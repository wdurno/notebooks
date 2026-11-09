"""Synthetic artifact fixtures exercise read-only reporting, not science."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest


def test_relative_artifact_root_is_stable_from_notebook_working_directory(monkeypatch, tmp_path):
    from numerical_experiment.picar import report
    monkeypatch.chdir(tmp_path)
    relative = Path("cache/numerical_experiment/picar/phase0")
    assert report.resolve_root(relative) == Path(report.__file__).resolve().parents[2] / relative

from numerical_experiment.picar import report
from numerical_experiment.picar.schemas import METRIC_COLUMNS, metric_row


def test_notebook_cells_execute_from_notebook_directory(tmp_path):
    code = (
        "import contextlib, io\n"
        "from numerical_experiment.picar.report import make_notebook\n"
        f"book = make_notebook({str(tmp_path)!r})\n"
        "namespace = {}\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    for cell in book.cells:\n"
        "        if cell.cell_type == 'code':\n"
        "            exec(compile(cell.source, '<notebook>', 'exec'), namespace)\n"
        "assert namespace['snapshot']['scientific_units'] == 0\n"
    )
    project = Path(report.__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, "-c", code], cwd=report.DEFAULT_NOTEBOOK.parent,
        env={**os.environ, "PYTHONPATH": str(project), "MPLCONFIGDIR": str(tmp_path / "mpl")},
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def seal(path):
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def unit(path, rows=None, *, smoke=True, version="fixture-v1", phase=0):
    path.mkdir(parents=True)
    config = {"config": {"schema_version": 1, "version": version,
                          "mode": "smoke" if smoke else "production", "model": {"kind": "tiny"}},
              "source_sha256": {}, "condition": "replay_ewc", "replica": 0}
    write_json(path / "config.json", config)
    write_json(path / "summary.json", {"phase": phase, "smoke_only": smoke})
    with (path / "metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows or [metric_row(phase="phase0_smoke", smoke_only=smoke)])
    manifest = {"artifact_schema_version": 1, "metric_schema_version": 1, "storage_schema_version": 1,
        "status": "completed", "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "final_files": {name: seal(path / name) for name in ("summary.json", "metrics.csv")}}
    write_json(path / "manifest.json", manifest)
    write_json(path / "COMPLETED", {"manifest_sha256": seal(path / "manifest.json")["sha256"]})
    return path


def panel_rows(condition, rates, *, budget=10, replica=0):
    return [metric_row(phase="phase4_fresh", condition=condition, replica=replica,
        new_transitions=n, step=i, episode=0, test_domain="B", test_success=p,
        test_count=20, smoke_only=False) for i, (n, p) in enumerate(zip((0, budget), rates))]


def test_pending_root_does_not_create_directories(tmp_path, monkeypatch):
    root = tmp_path / "does-not-exist"
    monkeypatch.setenv("PICAR_CACHE_ROOT", str(root))
    snapshot = report.load_report()
    assert snapshot["status"] == "pending" and not root.exists()
    assert [row["phase"] for row in report.progress_rows(snapshot)] == list(range(6))


def test_completed_unit_validates_nested_config_and_scalar_seals(tmp_path):
    path = unit(tmp_path / "bootstrap")
    loaded = report.read_completed_unit(path)
    assert loaded["config"]["mode"] == "smoke" and loaded["smoke_only"]
    assert loaded["metrics"][0]["phase"] == "phase0_smoke"
    (path / "metrics.csv").write_text((path / "metrics.csv").read_text() + "\n")
    with pytest.raises(report.ArtifactError, match="hash/size"):
        report.read_completed_unit(path)


def test_unknown_manifest_and_incomplete_units_fail_clearly_without_repair(tmp_path):
    path = unit(tmp_path / "unit")
    value = json.loads((path / "manifest.json").read_text())
    value["artifact_schema_version"] = 999
    write_json(path / "manifest.json", value)
    with pytest.raises(report.ArtifactError, match="unknown artifact"):
        report.load_report(tmp_path)
    (path / "COMPLETED").unlink()
    with pytest.raises(report.ArtifactError, match="incomplete"):
        report.read_completed_unit(path)
    assert not (path / "COMPLETED").exists()


def test_live_progress_ignores_partial_metrics_and_transaction_orphans(tmp_path):
    path = unit(tmp_path / "replica-000" / "stm_fd")
    (path / "COMPLETED").unlink()
    write_json(path / "progress.json", {"phase": 0, "step": 7, "new_transitions": 3,
                                       "target": 12, "test_success": 1.0})
    (path / "metrics.csv").write_text("partially flushed incompatible CSV")
    orphan = path / "transactions" / "orphan"
    orphan.mkdir(parents=True)
    write_json(orphan / "progress.json", {"made_up": "must not be read"})
    snapshot = report.load_report(tmp_path)
    assert snapshot["complete_units"] == [] and len(snapshot["live"]) == 1
    assert snapshot["live"][0]["counts"] == {"step": 7, "new_transitions": 3, "target": 12}
    assert "test_success" not in snapshot["live"][0]


def test_canonical_pointer_progress_wins_over_lagging_root_mirror(tmp_path):
    path = unit(tmp_path / "unit")
    (path / "COMPLETED").unlink()
    write_json(path / "progress.json", {"step": 1})
    transaction = path / "transactions" / "commit-2"
    transaction.mkdir(parents=True)
    write_json(transaction / "progress.json", {"phase": 0, "step": 2})
    write_json(transaction / "manifest.json", {"files": {"progress.json": seal(transaction / "progress.json")}})
    write_json(path / "CURRENT.json", {"storage_schema_version": 1, "commit_id": "commit-2",
                                       "manifest": seal(transaction / "manifest.json")})
    snapshot = report.load_report(tmp_path)
    assert snapshot["live"][0]["counts"] == {"step": 2}


def test_smoke_is_excluded_and_heldout_counts_not_training_success_are_aggregated(tmp_path):
    rows = panel_rows("replay_ewc", [.2, .4])
    unit(tmp_path / "smoke", rows, smoke=True, phase=4)
    assert report.heldout_curves(report.load_report(tmp_path)) == []
    scientific = tmp_path / "science"
    more = metric_row(phase=4, condition="replay_ewc", replica=0, new_transitions=10,
        step=3, episode=1, test_domain="B", test_success=1., test_count=10, smoke_only=False)
    training = metric_row(phase=4, condition="replay_ewc", replica=0, success=True, smoke_only=False)
    unit(scientific, [*rows, more, training], smoke=False, phase=4)
    curves = report.heldout_curves(report.load_report(tmp_path))
    assert curves[0]["samples"] == [0, 10]
    assert curves[0]["success"][-1] == pytest.approx((.4 * 20 + 10) / 30)
    assert curves[0]["panel_counts"] == [20, 30]


def test_paired_failure_and_unknown_versions_never_pool_silently(tmp_path):
    unit(tmp_path / "control", panel_rows("replay_ewc", [.2, .4]), smoke=False, phase=4)
    unit(tmp_path / "fd", panel_rows("stm_fd", [.2, .8]), smoke=False, phase=4)
    result = report.paired_failure_summary(report.load_report(tmp_path))
    assert result["mean"] == pytest.approx(.2) and result["n_pairs"] == 1
    assert result["ci95"] is None
    unit(tmp_path / "other-version", panel_rows("replay_ewc", [.1, .3], replica=1),
         smoke=False, version="different", phase=4)
    with pytest.raises(report.ArtifactError, match="multiple experiment versions"):
        report.heldout_curves(report.load_report(tmp_path))


def test_acceptance_is_visible_but_never_scientific(tmp_path):
    path = tmp_path / "qwen-acceptance-v1"
    path.mkdir()
    write_json(path / "manifest.json", {"schema_version": 1, "kind": "qwen_acceptance", "status": "completed"})
    write_json(path / "summary.json", {"schema_version": 1, "smoke_only": True, "load_seconds": 2.})
    write_json(path / "COMPLETED", {"schema_version": 1})
    snapshot = report.load_report(tmp_path)
    assert len(snapshot["diagnostics"]) == 1 and snapshot["scientific_units"] == 0
    assert report.progress_rows(snapshot)[0]["diagnostics"] == 1
    assert report.readiness_rows(snapshot)[0]["FD_convergence"] == "not certified"
    write_json(path / "summary.json", {"schema_version": 1, "fd_convergence_pass": False})
    assert report.readiness_rows(report.load_report(tmp_path))[0]["FD_convergence"] == "failed"
    write_json(path / "summary.json", {"schema_version": 1, "fd_convergence_pass": True})
    assert report.readiness_rows(report.load_report(tmp_path))[0]["FD_convergence"] == "passed"


@pytest.mark.parametrize("has_summary", [True, False])
def test_failed_qwen_diagnostic_remains_visible_without_completion_or_repair(tmp_path, has_summary):
    path = tmp_path / "qwen-acceptance-failed"
    path.mkdir()
    write_json(path / "manifest.json", {"schema_version": 1, "kind": "qwen_acceptance",
        "status": "failed", "error": "FD convergence did not pass"})
    if has_summary:
        write_json(path / "summary.json", {"schema_version": 1, "fd_convergence_pass": False})
    snapshot = report.load_report(tmp_path)
    assert snapshot["complete_units"] == [] and snapshot["scientific_units"] == 0
    assert len(snapshot["diagnostics"]) == 1
    row = report.readiness_rows(snapshot)[0]
    assert row["artifact"] == "failed, smoke only" and "FD convergence" in row["error_note"]
    assert row["FD_convergence"] == ("failed" if has_summary else "not certified")
    assert report.progress_rows(snapshot)[0]["diagnostic_failures"] == 1
    assert not (path / "COMPLETED").exists()
    assert (path / "summary.json").exists() == has_summary


def test_notebook_build_renders_progress_readiness_and_pending_without_kernel(tmp_path):
    unit(tmp_path / "root" / "bootstrap")
    path = report.build_notebook(tmp_path / "root", tmp_path / "report.ipynb")
    book = nbformat.read(path, as_version=4)
    nbformat.validate(book)
    assert book.metadata["plan19_snapshot"]["completed_units"] == 1
    code = [cell for cell in book.cells if cell.cell_type == "code"]
    assert all(cell.outputs for cell in code)
    assert "Phase 0 smoke" in str(code[0].outputs)
    assert any("text/html" in output.get("data", {}) for output in code[0].outputs)
    assert "Readiness measurements pending" in str(code[2].outputs)
    assert "not the full context-8/batch-4 workload" in "\n".join(c.source for c in book.cells)
    assert [c.id for c in report.make_notebook().cells] == [c.id for c in report.make_notebook().cells]
    assert "train(" not in "\n".join(cell.source for cell in code)
    assert not (tmp_path / "root" / "bootstrap" / "report.ipynb").exists()


def test_report_import_and_notebook_reload_need_no_torch_or_model_stack(tmp_path):
    env = {**os.environ, "PICAR_CACHE_ROOT": str(tmp_path / "pending"), "MPLBACKEND": "Agg"}
    code = ("import sys; from numerical_experiment.picar import report; "
        "book=report.make_notebook(); ns={}; "
        "[exec(compile(c.source,'<report>','exec'),ns) for c in book.cells if c.cell_type=='code']; "
        "assert not any(x in sys.modules for x in ('torch','transformers','peft','bitsandbytes')); "
        "assert 'src.continual.picar_math' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=30, capture_output=True)


def test_optional_video_uses_stored_synchronized_frames_and_no_overwrite(tmp_path):
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    Image = pytest.importorskip("PIL.Image")
    ego, topdown = tmp_path / "ego.png", tmp_path / "top.png"
    Image.new("RGB", (40, 40), "red").save(ego)
    Image.new("RGB", (40, 40), "green").save(topdown)
    frames = [{"time_s": at, "ego_path": ego, "topdown_path": topdown,
               "instruction": "fixture stop", "outcome": "fixture_failure"} for at in (0., .2)]
    output = tmp_path / "gallery.mp4"
    result = report.assemble_video(frames, output, seed=19, selection="failure_gallery", fps=5,
                                   panel_size=(160, 120))
    assert output.is_file() and result["selection"] == "failure_gallery"
    assert result["frames"] == 2 and result["topdown_reader_only"]
    with imageio.get_reader(output) as reader:
        assert reader.get_data(0).shape[:2] == (232, 320)
    with pytest.raises(FileExistsError):
        report.assemble_video(frames, output, seed=19)
    frames[1]["topdown_time_s"] = .5
    with pytest.raises(ValueError, match="not synchronized"):
        report.assemble_video(frames, tmp_path / "bad.mp4", seed=19)
