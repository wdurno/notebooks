"""Deterministic scalar fixtures: no model, profiler, network, or GPU."""

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest

from numerical_experiment.picar import phase1_report as phase1
from numerical_experiment.picar import report
from numerical_experiment.picar.schemas import METRIC_COLUMNS


RECIPE = dict(top_layers=1, hidden_size=64, ltm_rank=4, stm_rank=1, context_length=8,
              batch=1, fd_probes=1, fisher_draws=2, image_width=64, image_height=48,
              max_pixels=12544, max_tokens=512)


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def seal(path):
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def unit(root, name, kind, **fields):
    path = root / "phase1" / name
    path.mkdir(parents=True)
    config = {"config": {"schema_version": 1, "version": "phase1-fixture", "mode": "smoke"},
              "phase1": {"kind": kind, "recipe": fields.get("recipe", RECIPE)}}
    write_json(path / "config.json", config)
    write_json(path / "summary.json", {"phase": 1, "smoke_only": True, "kind": kind, **fields})
    with (path / "metrics.csv").open("w") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_COLUMNS).writeheader()
    write_json(path / "manifest.json", {"artifact_schema_version": 1, "metric_schema_version": 1,
        "storage_schema_version": 1, "status": "completed", "completed_at": "2026-11-09T12:00:00Z",
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "final_files": {name: seal(path / name) for name in ("summary.json", "metrics.csv")}})
    write_json(path / "COMPLETED", {"manifest_sha256": seal(path / "manifest.json")["sha256"]})
    return path


def probe(root, name="unit-probe", **fields):
    defaults = {"recipe": RECIPE, "status": "passed", "timings": {
        "load_seconds": 4., "inference_seconds": .5, "nll_backward_seconds": .7,
        "fd_backward_seconds": None, "fit_control_seconds": 1., "fit_fd_seconds": 2.,
        "fisher_seconds": 3., "teacher_seconds": None, "evaluator_seconds": .2},
        "memory": {"peak_allocated_bytes": 6 * 1024**3, "peak_reserved_bytes": 7 * 1024**3,
                   "total_bytes": 12 * 1024**3, "headroom_bytes": 5 * 1024**3}}
    return unit(root, name, "phase1_resource_probe", **{**defaults, **fields})


def forecast(root, **fields):
    defaults = {"selected_recipe": RECIPE, "phase_hours": {"phase2": [2, 4], "phase3": [3, 5],
        "phase4": [8, 12], "phase5": [.5, 1]}, "total_hours_range": [16, 25], "budgetfit": False,
        "qualifiedassumptions": ["Evaluation and sleep included; phase 2 will reforecast."],
        "blockers": ["Upper bound exceeds 20 h; no extension authorized."]}
    return unit(root, "unit-forecast", "phase1_forecast", **{**defaults, **fields})


def output_text(outputs):
    return "\n".join(o.get("text", "") + o.get("data", {}).get("text/plain", "") for o in outputs)


def test_append_inserts_prerendered_section_after_readiness_before_development(tmp_path):
    probe(tmp_path)
    forecast(tmp_path)
    snapshot = report.load_report(tmp_path)
    book = report._populate_outputs(report.make_notebook(tmp_path), snapshot)
    previous = {c.id: c for c in book.cells}
    assert phase1.append_phase1(book, snapshot) is book
    nbformat.validate(book)
    index = next(i for i, c in enumerate(book.cells) if c.source.startswith("## Phase 1"))
    assert book.cells[index - 1].metadata["report_section"] == "readiness"
    assert book.cells[index + 2].source.startswith("## Development")
    for cell in book.cells:
        if cell.id in previous:
            assert cell == previous[cell.id]
    cell = book.cells[index + 1]
    text = output_text(cell.outputs)
    assert "16" in text and "25" in text and "Recorded budget fit: false" in text
    assert "resource only" in text and "Evaluation and sleep included" in text
    assert "Selected recipe" in text and "unavailable" in text
    images = [o.data["image/png"] for o in cell.outputs if "image/png" in o.get("data", {})]
    assert len(images) == 2
    for data in images:
        assert base64.b64decode(data).startswith(b"\x89PNG\r\n\x1a\n")
    ids = [c.id for c in book.cells]
    phase1.append_phase1(book, snapshot)
    assert [c.id for c in book.cells] == ids
    assert snapshot["scientific_units"] == 0


@pytest.mark.parametrize("status,top_layers", [("oom", 1), ("inadmissible", 1), ("failed", 1), ("passed", 0)])
def test_nonadmission_and_top_zero_cannot_be_hidden_by_recorded_selection(tmp_path, status, top_layers):
    recipe = {**RECIPE, "top_layers": top_layers}
    probe(tmp_path, status=status, recipe=recipe, reason="Fixed admission failure", timings=None, memory=None)
    forecast(tmp_path, selected_recipe=recipe)
    text = output_text(phase1.phase1_outputs(report.load_report(tmp_path)))
    assert status in text and "Fixed admission failure" in text
    assert "resource admission remains unresolved" in text
    assert "unavailable" in text
    if top_layers == 0:
        assert "top-0 diagnostic only" in text


def test_missing_and_zero_timings_are_distinct_in_tables_and_plot(tmp_path):
    probe(tmp_path, timings={"inference_seconds": 0., "fit_control_seconds": None}, memory={})
    snapshot = report.load_report(tmp_path)
    text = output_text(phase1.phase1_outputs(snapshot))
    assert "Infer s" in text and "Control fit s" in text and "unavailable" in text
    import matplotlib.pyplot as plt
    figure = phase1.plot_resources(snapshot)
    try:
        collections = figure.axes[1].collections
        assert len(collections) == 1
        assert float(collections[0].get_offsets()[0][0]) == 0.
        assert "VRAM unavailable" in [t.get_text() for t in figure.axes[0].texts]
    finally:
        plt.close(figure)


def test_missing_forecast_is_pending_not_plan_prior_or_summed_probe_runtime(tmp_path):
    probe(tmp_path, active_seconds=1000.)
    probe(tmp_path, "unit-second", active_seconds=1000.)
    text = output_text(phase1.phase1_outputs(report.load_report(tmp_path)))
    assert "Overall forecast pending" in text and "not a measured runtime promise" in text
    assert "Selected recipe: not recorded" in text
    assert "1000" not in text and "2000" not in text
    assert "Phase 0" not in text


def test_nested_scene_roles_display_plain_numbers_and_honest_limits(tmp_path):
    unit(tmp_path, "unit-scene", "phase1_scene_audit", scene={"geometry": {"passed": True},
         "physical_calibration": {"status": "not measured"}, "controller": {"successes": 3, "layouts": 4}})
    unit(tmp_path, "unit-roles", "phase1_roles", teacher={"success_rate": .75, "fallback_count": 2},
         student={"status": "not evaluated"}, evaluator={"reward_check_pass": False})
    text = output_text(phase1.phase1_outputs(report.load_report(tmp_path)))
    assert "0.75" in text and "fallback count" in text and "2" in text
    assert "not evaluated" in text and "false" in text and "not measured" in text
    assert "not held-out student competence" in text and "physical calibration" in text


def test_incomplete_payloads_are_not_opened_or_repaired(tmp_path, monkeypatch):
    path = probe(tmp_path)
    (path / "COMPLETED").unlink()
    write_json(path / "progress.json", {"phase": 1, "counts": {"step": 2}})
    (path / "summary.json").write_text("UNFLUSHED SECRET")
    (path / "metrics.csv").write_text("UNFLUSHED METRICS")
    snapshot = report.load_report(tmp_path)
    assert snapshot["live"] and not snapshot["complete_units"]

    def no_read(*args, **kwargs):
        raise AssertionError("Phase 1 report must consume the snapshot, not read paths")

    monkeypatch.setattr(Path, "open", no_read)
    text = output_text(phase1.phase1_outputs(snapshot))
    assert "Resource probes pending" in text and "UNFLUSHED" not in text
    assert not (path / "COMPLETED").exists()


def test_many_candidates_are_bounded_and_keep_failure_visibility(tmp_path):
    for i in range(30):
        probe(tmp_path, f"unit-{i:03d}", recipe={**RECIPE, "hidden_size": 64 + i})
    probe(tmp_path, "unit-old-oom", status="oom", reason="fixture out of memory")
    forecast(tmp_path)
    text = output_text(phase1.phase1_outputs(report.load_report(tmp_path)))
    assert "Showing 6 of 31" in text and "fixture out of memory" in text
    assert "H=64" in text and "H=70" not in text


def test_repeated_selected_probes_do_not_crowd_out_failures(tmp_path):
    for i in range(8):
        probe(tmp_path, f"unit-{i:03d}")
    probe(tmp_path, "unit-oom", status="oom", recipe={**RECIPE, "top_layers": 4})
    forecast(tmp_path)
    text = output_text(phase1.phase1_outputs(report.load_report(tmp_path)))
    assert "oom" in text and "Showing 6 of 9" in text


def test_failed_numerical_checks_are_visible_even_if_probe_claims_passed(tmp_path):
    probe(tmp_path, metricschecks={"nll_finite": True, "fd_convergence_pass": False})
    forecast(tmp_path)
    text = output_text(phase1.phase1_outputs(report.load_report(tmp_path)))
    assert "passed; checks failed" in text and "fd_convergence_pass=false" in text
    assert "resource admission remains unresolved" in text


def test_aggregate_competence_and_denominator_not_displaced_by_per_scene_details(tmp_path):
    unit(tmp_path, "unit-scene", "phase1_scene_audit", successes=3, scene_count=4, all_feasible=False,
         scene_outcomes=[{"geometry": {"clearance": .1}, "success": True}] * 100,
         limitations=["Privileged controller is not student competence."])
    text = output_text(phase1.phase1_outputs(report.load_report(tmp_path)))
    assert "successes" in text and "scene count" in text and "all feasible" in text
    assert "Privileged controller is not student competence" in text
    assert "scene outcomes" not in text


def test_critical_role_numbers_are_not_displaced_by_many_reward_checks(tmp_path):
    unit(tmp_path, "unit-roles", "phase1_roles", teacher={"success_rate": .75, "episode_count": 4,
         "fallback_count": 1}, evaluator={f"reward_check_{i:02d}": True for i in range(30)})
    text = output_text(phase1.phase1_outputs(report.load_report(tmp_path)))
    assert "teacher.success rate" in text and "teacher.episode count" in text
    assert "teacher.fallback count" in text and "in archived summary" in text


def test_profiler_checks_alias_and_metadata_recipe_are_supported(tmp_path):
    unit(tmp_path, "unit-probe", "phase1_resource_probe", status="passed", checks={"finite_action": False})
    forecast(tmp_path)
    text = output_text(phase1.phase1_outputs(report.load_report(tmp_path)))
    assert "L=1 H=64 C=8 B=1" in text and "finite_action=false" in text
    assert "resource admission remains unresolved" in text


def test_archived_artifact_link_is_relative_to_requested_notebook_directory(tmp_path):
    snapshot = report.load_report(tmp_path / 'cache & artifacts')
    outputs = phase1.phase1_outputs(snapshot, notebook_dir=tmp_path / "notebooks")
    markup = "\n".join(o.get("data", {}).get("text/html", "") for o in outputs)
    assert 'href="../cache%20%26%20artifacts"' in markup
    assert "cache &amp; artifacts" in markup


def test_forecast_partial_ranges_are_unavailable_not_zero_or_extrapolated(tmp_path):
    forecast(tmp_path, phase_hours={"phase2": [2, 4], "phase3": None,
                                   "phase4": [12, 8], "phase5": [True, 1]}, total_hours_range=None)
    snapshot = report.load_report(tmp_path)
    rows = phase1._forecast_rows(snapshot)
    assert rows[0]["Low h"] == 2
    assert all(row["Low h"] is None for row in rows[1:])
    assert "unavailable" in output_text(phase1.phase1_outputs(snapshot))


def test_both_scope_forecasts_are_visible_without_summing_them(tmp_path):
    forecast(tmp_path, pairs=6, transitions_per_arm=2400)
    unit(tmp_path, "unit-forecast-small", "phase1_forecast", selected_recipe=RECIPE,
         pairs=4, transitions_per_arm=1200, total_hours_range=[10, 15], budgetfit=True)
    rows = phase1._scope_rows(report.load_report(tmp_path))
    assert {r["New transitions / arm"] for r in rows} == {1200, 2400}
    assert {r["Total active hours"] for r in rows} == {"16.0--25.0", "10.0--15.0"}


def test_phase1_conclusion_reports_blockers_without_claiming_learning(tmp_path):
    forecast(tmp_path, pairs=4, transitions_per_arm=1200)
    unit(tmp_path, "unit-campaign", "phase1_campaign", selected_recipe=RECIPE,
         blockers=["Teacher readiness unresolved."])
    snapshot = report.load_report(tmp_path)
    text = phase1.phase1_conclusion(snapshot)
    assert "16.0--25.0 h" in text and "Teacher readiness unresolved" in text
    assert "later learning phases have not started" in text
    assert "No sample-efficiency result is claimed" in text


def test_report_and_rerun_do_not_import_torch_profiler_or_model_stack(tmp_path):
    probe(tmp_path)
    forecast(tmp_path)
    project = Path(report.__file__).resolve().parents[2]
    code = (
        "import contextlib, io, sys\n"
        "from numerical_experiment.picar import report, phase1_report\n"
        f"snapshot = report.load_report({str(tmp_path)!r})\n"
        "book = phase1_report.append_phase1(report.make_notebook(), snapshot)\n"
        "namespace = {'snapshot': snapshot}\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    for cell in book.cells:\n"
        "        if cell.cell_type == 'code' and cell.metadata.get('plan19_phase1'):\n"
        "            exec(compile(cell.source, '<phase1>', 'exec'), namespace)\n"
        "assert not any(x in sys.modules for x in ('torch', 'transformers', 'peft', 'bitsandbytes', "
        "'numerical_experiment.picar.phase1', 'src.continual.picar_math'))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=report.DEFAULT_NOTEBOOK.parent,
        env={**os.environ, "PYTHONPATH": str(project), "MPLBACKEND": "Agg",
             "MPLCONFIGDIR": str(tmp_path / "mpl")}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_plot_png_contains_nonblank_raster(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    probe(tmp_path)
    outputs = phase1.phase1_outputs(report.load_report(tmp_path))
    image = next(o.data["image/png"] for o in outputs if "image/png" in o.get("data", {}))
    raster = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
    assert raster.width >= 800 and raster.height >= 300
    assert any(lo < hi for lo, hi in raster.getextrema())
