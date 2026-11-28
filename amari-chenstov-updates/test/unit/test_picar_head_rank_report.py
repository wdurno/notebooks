"""Small scalar fixtures for the read-only Fix 5A notebook reporter."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest

from numerical_experiment.picar_head_rank import report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))
    return path


def progress(root, **extra):
    return write(root / "progress.json", {
        "version": report.VERSION, "status": "running", "stage": "resource_audit",
        "charged_seconds": 720., "completed_units": 0, "planned_units": 12,
        "active_progress": {"n": 23}, "forecast": {"remaining_hours": [10., 13.]}, **extra,
    })


def panel(value=1., *, n=1024, environment="initial_A", count=8, success=False, **extra):
    return {"n": n, "environment": environment, "acquired_count": n + 3,
            "episodes": [{"raw_return": value, "success": success, "seed": i,
                          "videos": []} for i in range(count)], **extra}


def unit(root, name, *, method="td_replay", seed=199101, phase="confirmation", rank=64,
         source=False, completed=True, operation="trajectory", **extra):
    settings = {"method": method, "seed": seed, "phase": phase, "ewc_rank": rank,
                "stm_rank": 1, "total": 1024, "recipe": {"rank": rank, "ridge": .1,
                "fd_strength": 12.142951965332031 if method == "td_fd" else 0.}}
    settings = extra.pop("settings", settings)
    value = {"version": report.SOURCE_VERSION if source else report.VERSION, "schema_version": 1,
             "kind": ("head_awac_" if source else "head_rank_") + operation,
             "operation": operation, "settings": settings, "active_seconds": 30., **extra}
    path = root / "units" / name
    write(path / "summary.json", value)
    if source:
        write(path / "config.json", {"operation": operation, "settings": settings})
    if completed:
        (path / "COMPLETED").touch()
    return path


def source_config(root, source, **extra):
    return write(root / "orchestration/config.json", {"source": str(source), **extra})


def book(path):
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Plan 19 history", id="history-title"),
        nbformat.v4.new_code_cell("print('old')", execution_count=31,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="frozen\n")],
            metadata={"plan19_phase3_fix5": True, "annotation": [1, 2]},
            id="plan19-phase3-fix5a-progress"),
        nbformat.v4.new_markdown_cell("Unrelated user history", id="user-history"),
    ], metadata={"user": {"keep": True}})
    value = json.loads(nbformat.writes(notebook))
    write(path, value)
    return value


def rendered(outputs):
    return "\n".join(o["data"].get("text/plain", "") for o in outputs)


@pytest.fixture(autouse=True)
def no_canonical_notebook(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "DEFAULT_NOTEBOOK", tmp_path / "missing.ipynb")


def test_zero_units_and_absent_root_build_without_runtime(tmp_path):
    root, output = tmp_path / "absent", tmp_path / "report.ipynb"
    snapshot = report.load_report(root)
    assert snapshot["units"] == [] and not snapshot["complete"]
    assert "Not started / incomplete" in rendered(report.progress_outputs(snapshot))
    assert report.build_notebook(root, output) == output
    assert not root.exists()
    loaded = nbformat.read(output, as_version=4)
    nbformat.validate(loaded)
    assert "Provisional / incomplete" in loaded.cells[-1].source


def test_preserves_all_history_and_latest_output_even_with_explicit_base(tmp_path):
    root, output = tmp_path / "run", tmp_path / "report.ipynb"
    original = book(root / "history.ipynb")
    progress(root)
    report.build_notebook(root, output, root / "history.ipynb")
    first = json.loads(output.read_text())
    assert first["metadata"] == original["metadata"]
    assert [c for c in first["cells"] if report._TAG not in c["metadata"]] == original["cells"]
    assert first["cells"][1]["metadata"][report._TAG]
    assert len({c["id"] for c in first["cells"]}) == len(first["cells"])
    added = dict(nbformat.v4.new_markdown_cell("Added during active run", id="later-user-cell"))
    first["cells"].append(added)
    write(output, first)
    progress(root, charged_seconds=1440.)
    report.build_notebook(root, output, root / "history.ipynb")
    refreshed = json.loads(output.read_text())
    assert [c for c in refreshed["cells"] if report._TAG not in c["metadata"]] == original["cells"] + [added]
    assert len(refreshed["cells"]) == len(first["cells"])
    assert json.loads((root / "history.ipynb").read_text()) == original
    assert "0.4" in rendered(refreshed["cells"][1]["outputs"])


def test_only_sealed_summaries_are_opened_and_live_progress_remains_preview(tmp_path, monkeypatch):
    path = unit(tmp_path, "active", completed=False)
    (path / "summary.json").write_text("not JSON")
    progress(tmp_path)
    opened = []
    original = report._read

    def read(path, limit):
        opened.append(Path(path))
        return original(path, limit)

    monkeypatch.setattr(report, "_read", read)
    snapshot = report.load_report(tmp_path)
    assert opened == [tmp_path / "progress.json"]
    text = rendered(report.progress_outputs(snapshot))
    assert "23" in text and "unsealed preview" in text and "0 / 12" in text
    assert "10.0" in text and not snapshot["complete"]


def test_active_preview_reads_only_small_configuration_and_does_not_enter_comparisons(tmp_path):
    path = unit(tmp_path, "active", completed=False)
    (path / "summary.json").write_text("unsealed; must not be read")
    write(path / "config.json", {"version": report.VERSION, "settings": {
        "phase": "development", "seed": 199301, "method": "td_replay", "total": 1024}})
    progress(tmp_path, active_unit=str(path), active_progress={"n": 525, "evaluation_preview": [
        {"n": 512, "environment": "initial_A", "episodes": 2, "success": 1., "raw_return": 80.9}]})
    snapshot = report.load_report(tmp_path)
    text = rendered(report.progress_outputs(snapshot))
    assert "Live Held-Out Preview (Unsealed)" in text and "80.9" in text
    assert snapshot["units"] == [] and report.evaluation_rows(snapshot) == []
    assert snapshot["active_settings"]["method"] == "td_replay"


def test_remaining_forecast_uses_new_completed_paths_and_excludes_reused_controls(tmp_path):
    progress(tmp_path)
    unit(tmp_path, "td-dev", phase="development", seed=199301, active_seconds=3600.)
    unit(tmp_path, "admission", operation="rank_admission", rank=64)
    snapshot = report.load_report(tmp_path)
    first = report.remaining_forecast(snapshot)
    assert first is not None and first[0] > 5
    unit(tmp_path, "fd-dev", method="td_fd", phase="development", seed=199301, active_seconds=3600.)
    snapshot = report.load_report(tmp_path)
    second = report.remaining_forecast(snapshot)
    assert second[0] < first[0] and second[1] < first[1]
    snapshot["active_settings"] = {"phase": "confirmation", "seed": 199101,
                                   "method": "td_replay", "total": 1024}
    snapshot["progress"]["active_progress"] = {"n": 512}
    third = report.remaining_forecast(snapshot)
    assert third[0] < second[0] and third[1] < second[1]
    snapshot["complete"] = True
    assert report.remaining_forecast(snapshot) == [0., 0.]


def test_source_configs_filter_before_opening_large_unrelated_summaries(tmp_path, monkeypatch):
    root, source = tmp_path / "new", tmp_path / "source"
    source_config(root, source)
    wanted = unit(source, "td-control", rank=0, source=True)
    for name, kwargs in (("mc", {"method": "mc_replay"}), ("rank10", {"rank": 10}),
                         ("pilot", {"phase": "resource"})):
        path = unit(source, name, source=True, **kwargs)
        (path / "summary.json").write_text("unrelated summary must not be opened")
    (source / "frames.pt").write_text("never read")
    opened, original = [], report._read

    def read(path, limit):
        opened.append(Path(path))
        return original(path, limit)

    monkeypatch.setattr(report, "_read", read)
    snapshot = report.load_report(root)
    assert [u["name"] for u in snapshot["controls"]] == ["td-control"]
    summaries = [p for p in opened if p.name == "summary.json"]
    assert summaries == [wanted / "summary.json"]
    assert all(p.suffix == ".json" for p in opened)


def test_source_receipt_detects_changes_without_touching_source(tmp_path):
    root, source = tmp_path / "new", tmp_path / "source"
    path = unit(source, "control", rank=0, source=True)
    seal = hashlib.sha256((path / "summary.json").read_bytes()).hexdigest()
    source_config(root, source, source_receipts=[{"unit": str(path), "summary_sha256": seal}])
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    report.build_notebook(root, tmp_path / "out.ipynb")
    assert before == {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    source_config(root, source, source_receipts=[{"unit": str(path), "summary_sha256": "0" * 64}])
    with pytest.raises(report.ArtifactError, match="checksum"):
        report.load_report(root)


@pytest.mark.parametrize("field", ["rank", "realized_rank", "factor_columns", "eigenvalues", "factor_eigenvalues"])
def test_archive_bounds_accept_64_reject_65(tmp_path, field):
    value = list(range(64)) if "eigenvalues" in field else 64
    path = unit(tmp_path, "rank64", archive={"rank": 64, field: value})
    assert report.load_report(tmp_path)["units"]
    summary = json.loads((path / "summary.json").read_text())
    summary["archive"][field] = list(range(65)) if "eigenvalues" in field else 65
    write(path / "summary.json", summary)
    with pytest.raises(report.ArtifactError, match="64"):
        report.load_report(tmp_path)


@pytest.mark.parametrize("where", ["archive_metrics", "diagnostics", "allocation", "final_states"])
def test_nested_scalar_schemas_are_bounded(tmp_path, where):
    if where == "archive_metrics":
        extra = {where: [{"n": 5, "rank": 64, "factor_eigenvalues": [1.] * 65}]}
    elif where == "diagnostics":
        extra = {"operation": "initial_archives", "diagnostics": {"ranks": {"64": {"rank": 65}}}}
    elif where == "allocation":
        extra = {"allocation": {"critic": {"factor_column_energy": [1.] * 65}}}
    else:
        extra = {"final_states": [[0.] * 513]}
    unit(tmp_path, "bad", **extra)
    with pytest.raises(report.ArtifactError):
        report.load_report(tmp_path)


def test_new_summary_does_not_accept_old_version_or_boolean_schema(tmp_path):
    path = unit(tmp_path, "bad", version=report.SOURCE_VERSION)
    with pytest.raises(report.ArtifactError, match="version"):
        report.load_report(tmp_path)
    value = json.loads((path / "summary.json").read_text())
    value.update(version=report.VERSION, schema_version=True)
    write(path / "summary.json", value)
    with pytest.raises(report.ArtifactError, match="schema_version"):
        report.load_report(tmp_path)


@pytest.mark.parametrize("content", ['{"value": NaN}', '{"value": 1e999}', '[]'])
def test_invalid_json_fails_clearly(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_text(content)
    with pytest.raises(report.ArtifactError):
        report._json(path)


def test_size_bounds_fail_before_overwriting_output(tmp_path, monkeypatch):
    root, output = tmp_path / "run", tmp_path / "out.ipynb"
    progress(root)
    original = book(output)
    monkeypatch.setattr(report, "MAX_TOTAL_BYTES", 1)
    with pytest.raises(report.ArtifactError, match="total"):
        report.build_notebook(root, output)
    assert json.loads(output.read_text()) == original
    with pytest.raises(report.ArtifactError, match="size bound"):
        report._read(root / "progress.json", 1)


def test_completed_run_and_source_output_are_protected(tmp_path):
    root, source = tmp_path / "new", tmp_path / "source"
    source_config(root, source)
    source.mkdir()
    with pytest.raises(report.ArtifactError, match="read-only"):
        report.build_notebook(root, source / "new.ipynb")
    completed = tmp_path / "sealed"
    completed.mkdir()
    (completed / "COMPLETED").touch()
    with pytest.raises(report.ArtifactError, match="completed"):
        report.build_notebook(root, completed / "new.ipynb")


def four_conditions(tmp_path, *, second=True):
    root, source = tmp_path / "new", tmp_path / "source"
    source_config(root, source)
    for seed, values in ((199101, (1., 3., 4., 10.)), (199102, (2., 6., 7., 14.))):
        if seed == 199102 and not second:
            continue
        for condition, value in zip(("T0", "F0", "T+", "F+"), values):
            reused = condition.endswith("0")
            unit(source if reused else root, f"{condition}-{seed}", source=reused,
                 rank=0 if reused else 64, method="td_fd" if condition.startswith("F") else "td_replay",
                 seed=seed, n=1031, timings={"fitting": 2.},
                 evaluations=[panel(value, environment=e) for e in report.ENVIRONMENTS])
    return root, source


def test_paired_effects_are_within_seed_and_interaction_has_correct_sign(tmp_path):
    root, source = four_conditions(tmp_path)
    # Deliberately enormous development values must not affect assessment.
    unit(root, "development", phase="development", evaluations=[panel(99999.)])
    rows = report.paired_endpoint_rows(report.load_report(root))
    rows = {r["Contrast"]: r for r in rows if r["Environment"] == "moderate_A" and r["Metric"] == "Raw return"}
    expected = {"Rank, TD: T+ - T0": (3., 5.), "Rank, FD: F+ - F0": (7., 8.),
                "FD, diagonal: F0 - T0": (2., 4.), "FD, correlated: F+ - T+": (6., 7.),
                "Interaction: (F+ - T+) - (F0 - T0)": (4., 3.)}
    for name, (first, second) in expected.items():
        row = rows[name]
        assert row["199101"] == first and row["199102"] == second
        assert row["Mean gap"] == (first + second) / 2
        assert row["Seed range (not CI)"] == [min(first, second), max(first, second)]
        assert row["Pairs"] == 2
        assert row["Actual acquisitions"]["199101"]
        assert row["Fitting s"]["199101"]


def test_missing_seed_is_pending_not_zero(tmp_path):
    root, _ = four_conditions(tmp_path, second=False)
    rows = report.paired_endpoint_rows(report.load_report(root))
    row = next(r for r in rows if r["Contrast"] == "Rank, TD: T+ - T0" and r["Metric"] == "Raw return")
    assert row["199101"] == 3 and row["199102"] is None
    assert row["Mean gap"] == 3 and row["Pairs"] == 1 and row["Status"] == "Pending pairs"


def test_complete_conclusion_explains_interaction_without_mistaking_it_for_fd_lift(tmp_path):
    root, source = four_conditions(tmp_path)
    for directory, values in ((source, {"T0": 10., "F0": 8.}),
                              (root, {"T+": 8., "F+": 8.})):
        for path in (directory / "units").glob("*/summary.json"):
            value = json.loads(path.read_text())
            condition = path.parent.name.split("-")[0]
            for p in value["evaluations"]:
                for e in p["episodes"]:
                    e["raw_return"] = values[condition]
            write(path, value)
    snapshot = report.load_report(root)
    snapshot["complete"] = True
    text = report.conclusion(snapshot)
    assert "Scope complete" in text and "No demonstrated FD acquisition benefit" in text
    assert "| TD + FD, rank 64 | 0/16 | 0/16 | 8.000 |" in text
    assert "reduced TD performance, not improved FD performance" in text
    assert "not 16 independent replicas" in text and "not confidence intervals" in text
    assert "No samples-saved/S estimate" in text


def test_missing_pairs_cannot_produce_final_conclusion(tmp_path):
    root, _ = four_conditions(tmp_path, second=False)
    snapshot = report.load_report(root)
    snapshot["complete"] = True
    text = report.conclusion(snapshot)
    assert "Provisional / incomplete" in text and "remains pending" in text
    assert "Scope complete" not in text


def test_plot_panel_counts_are_visible_captions_not_only_image_alt_text(tmp_path):
    unit(tmp_path, "one", evaluations=[panel(1., n=0, count=4), panel(2.)])
    outputs = report.result_outputs(report.load_report(tmp_path))
    images = [i for i, output in enumerate(outputs) if "image/png" in output.get("data", {})]
    assert images
    for index in images:
        assert "<p>" in outputs[index+1]["data"]["text/html"]
    caption = outputs[images[0]+1]["data"]["text/html"]
    assert "common first four layouts" in caption and "tables use all eight layouts" in caption


@pytest.mark.parametrize("mismatch", ["midpoint", "budget", "layout", "missing", "duplicate"])
def test_unmatched_or_missing_endpoints_do_not_produce_a_rank_effect(tmp_path, mismatch):
    root, _ = four_conditions(tmp_path, second=False)
    path = root / "units/T+-199101/summary.json"
    value = json.loads(path.read_text())
    if mismatch == "midpoint":
        for p in value["evaluations"]:
            p["n"] = 512
    elif mismatch == "budget":
        value["settings"]["total"] = 512
    elif mismatch == "layout":
        value["evaluations"][0]["episodes"][0]["seed"] = 999
    elif mismatch == "missing":
        value["evaluations"][0]["episodes"][0]["raw_return"] = None
    else:
        unit(root, "duplicate", evaluations=value["evaluations"])
    write(path, value)
    rows = report.paired_endpoint_rows(report.load_report(root))
    row = next(r for r in rows if r["Environment"] == "initial_A" and r["Metric"] == "Raw return"
               and r["Contrast"] == "Rank, TD: T+ - T0")
    assert row["199101"] is None and row["Mean gap"] is None and row["Pairs"] == 0


def test_common_four_curves_are_equal_seed_means_not_episode_pooling(tmp_path):
    unit(tmp_path, "one", evaluations=[panel(0., count=4)])
    p = panel(8., count=8)
    for e in p["episodes"][4:]:
        e["raw_return"] = 999.
    unit(tmp_path, "two", seed=199102, evaluations=[p])
    row = report.evaluation_rows(report.load_report(tmp_path))[0]
    assert row["raw_return"] == 4. and row["Layouts per trajectory"] == 4
    assert row["Return seed range"] == [0., 8.] and row["Trajectories"] == 2


def test_development_retention_requires_expanded_eight_layout_reference(tmp_path):
    root, source = tmp_path / "new", tmp_path / "source"
    source_config(root, source)
    control = unit(source, "old-dev", source=True, rank=0, phase="development", seed=199301,
                   evaluations=[panel(99., count=2)])
    unit(root, "positive-dev", phase="development", seed=199301,
         evaluations=[panel(12.), panel(8., environment="moderate_A")])
    rows = report.development_rows(report.load_report(root))
    assert all(r["Retention gap vs expanded T0"] is None for r in rows)
    unit(root, "expanded", operation="evaluation_expansion", phase="development_expansion", rank=0,
         source_unit=str(control), evaluations=[panel(10.), panel(3., environment="moderate_A")])
    rows = report.development_rows(report.load_report(root))
    assert len(rows) == 2
    positive = next(r for r in rows if r["Condition"] == "T+")
    assert positive["Retention gap vs expanded T0"] == 2.
    assert positive["Within +/-5"] is True


def test_profile_unit_types_and_64_direction_allocation_render(tmp_path, monkeypatch):
    allocation = {"actor": {"trace": 2., "correlated_trace_fraction": .99, "factor_column_energy": [1.] * 64},
                  "critic": {"trace": 3., "correlated_trace_fraction": .97, "factor_column_energy": [1.] * 64}}
    archive = {"rank": 64, "factor_eigenvalues": list(range(64)), "allocation": allocation}
    unit(tmp_path, "initial", operation="initial_archives", settings={},
         diagnostics={"ranks": {"64": archive}, "records": 515})
    unit(tmp_path, "resource", operation="resource_audit", settings={}, result={"healthy_ranks": [64],
         "timings": [{"rank": 64, "method": "td_fd", "fit_seconds": .5, "solve_seconds": .01}]})
    unit(tmp_path, "admit", operation="rank_admission", settings={}, rank=64, reason="healthy and fits budget")
    figures = []
    monkeypatch.setattr(report, "_figure", lambda fig, description:
                        (figures.append(fig) or {"data": {"text/plain": description}}))
    text = rendered(report.result_outputs(report.load_report(tmp_path)))
    assert "healthy and fits budget" in text and "0.97" in text and "Cached Resource Audit" in text
    assert list(figures[0].axes[0].lines[0].get_ydata()) == list(reversed(range(64)))


def test_missing_diagnostics_plot_gaps_instead_of_zeros(tmp_path, monkeypatch):
    unit(tmp_path, "new", metrics=[{"n": 1, "fd": .2, "ewc": 0., "post_step_ewc": .4},
                                   {"n": 2, "fd": None, "ewc": 0., "post_step_ewc": None}])
    figures = []
    monkeypatch.setattr(report, "_figure", lambda fig, description:
                        (figures.append(fig) or {"data": {"text/plain": description}}))
    report.diagnostic_outputs(report.load_report(tmp_path))
    values = figures[0].axes[0].lines[0].get_ydata()
    assert values[0] == .4 and values[1] != values[1]
    assert not figures[0].axes[2].lines


def test_initial_video_resolves_against_original_source_unit(tmp_path):
    original = tmp_path / "source/units/initial"
    original.mkdir(parents=True)
    (original / "initial.mp4").touch()
    p = panel(2., n=0, shared_initial_source=str(original))
    p["episodes"][0]["videos"] = [{"path": "initial.mp4"}]
    unit(tmp_path / "new", "one", evaluations=[p])
    unit(tmp_path / "new", "two", seed=199102, evaluations=[p])
    text = rendered(report.recording_outputs(report.load_report(tmp_path / "new"), tmp_path))
    assert text.count("initial.mp4") == 1 and "Shared initialization" in text
    assert "source/units/initial/initial.mp4" in text


def test_outcome_selected_success_video_is_linked_and_qualified(tmp_path):
    root = tmp_path / "extension"
    p = panel(environment="moderate_A")
    p["episodes"][3].update(success=True, episode_id="saved-success")
    original = unit(root, "one", evaluations=[p])
    gallery = root.with_name(root.name + "-success-example-v1")
    example = {"version": "plan19-saved-success-example-v1", "success": True,
               "new_evaluation": False, "source_unit": str(original), "n": 1024,
               "episode_id": "saved-success", "video": "camera.mp4", "seed": 199101,
               "raw_return": 81., "direct_path_blocked": False}
    write(gallery / "summary.json", example)
    summary_hash = hashlib.sha256((gallery / "summary.json").read_bytes()).hexdigest()
    write(gallery / "manifest.json", {"final_files": {"summary.json": {"sha256": summary_hash}}})
    write(gallery / "COMPLETED", {"manifest_sha256": hashlib.sha256((gallery / "manifest.json").read_bytes()).hexdigest()})
    (gallery / "camera.mp4").touch()
    outputs = report.recording_outputs(report.load_report(root), tmp_path)
    text = rendered(outputs)
    assert "Outcome-selected success" in text and "not representative" in text
    assert "camera.mp4" in text and "no top-down recording" in text
    assert '<a href="extension-success-example-v1/camera.mp4">video</a>' in outputs[0]["data"]["text/html"]
    example["episode_id"] = "not-a-saved-episode"
    write(gallery / "summary.json", example)
    with pytest.raises(report.ArtifactError, match="checksum mismatch"):
        report.recording_outputs(report.load_report(root), tmp_path)


def test_import_and_notebook_generation_do_not_import_torch_or_runtime(tmp_path):
    script = """
import importlib.abc
import sys
from pathlib import Path
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith(('torch.', 'numerical_experiment.picar.runtime',
                'numerical_experiment.picar_head_rank.campaign', 'numerical_experiment.picar_head_rank.numerics',
                'numerical_experiment.picar_head_awac')):
            raise AssertionError('runtime import: ' + fullname)
sys.meta_path.insert(0, Guard())
from numerical_experiment.picar_head_rank import report
report.DEFAULT_NOTEBOOK = Path(sys.argv[1]) / 'absent.ipynb'
report.build_notebook(Path(sys.argv[1]) / 'empty', Path(sys.argv[1]) / 'out.ipynb')
assert 'torch' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
