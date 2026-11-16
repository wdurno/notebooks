"""Small sealed scalar fixtures; reporting never runs the experiment."""

import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
import pytest

from numerical_experiment.picar import report as shared
from numerical_experiment.picar.schemas import METRIC_COLUMNS
from numerical_experiment.picar_initialization import report


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def seal(path):
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def unit(root, kind, *, name=None, **fields):
    path = root / "phase3" / report.CAMPAIGN / "units" / (name or kind)
    path.mkdir(parents=True)
    config = {"config": {"schema_version": 1, "version": report.VERSION, "mode": "production"}}
    write_json(path / "config.json", config)
    write_json(path / "summary.json", {"schema_version": 1, "kind": "initialization_" + kind,
        "phase": 3, "smoke_only": True, "development_only": True, **fields})
    with (path / "metrics.csv").open("w") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_COLUMNS).writeheader()
    manifest = {"artifact_schema_version": 1, "metric_schema_version": 1, "storage_schema_version": 1,
        "status": "completed", "completed_at": f"2026-11-13T12:00:{len(list(path.parent.iterdir())):02d}Z",
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "final_files": {key: seal(path / key) for key in ("summary.json", "metrics.csv")}}
    write_json(path / "manifest.json", manifest)
    write_json(path / "COMPLETED", {"manifest_sha256": seal(path / "manifest.json")["sha256"]})
    return path


def live(root, progress, *, name="orchestration", schema=True):
    path = root / "phase3" / report.CAMPAIGN / name
    path.mkdir(parents=True)
    write_json(path / "config.json", {"schema_version": 1, "version": report.VERSION})
    write_json(path / "manifest.json", {"artifact_schema_version": 1, "metric_schema_version": 1,
        "status": "running"})
    value = {"phase": 3, "version": report.VERSION, **({"schema_version": 1} if schema else {}), **progress}
    transaction = path / "transactions" / "epoch-2"
    transaction.mkdir(parents=True)
    write_json(transaction / "progress.json", value)
    write_json(transaction / "manifest.json", {"files": {"progress.json": seal(transaction / "progress.json")}})
    write_json(path / "CURRENT.json", {"storage_schema_version": 1, "commit_id": "epoch-2",
        "manifest": seal(transaction / "manifest.json")})
    write_json(path / "progress.json", {"schema_version": 1, "phase": 3, "status": "STALE mirror must not appear"})
    return path


def curves():
    return [{"stage": stage, "epoch": epoch, "optimization_exposures": epoch * 256,
        "train": {"joint_nll": 5. - epoch, "teacher_kl": .8 - epoch * .1, "actor_nll": 3. - epoch, "critic_nll": 2.},
        "validation": {"joint_nll": 5.5 - epoch * .5, "teacher_kl": .9 - epoch * .1,
                       "actor_nll": 3.5 - epoch * .5, "critic_nll": 2.},
        "diagnostics": {"gradient_norm_mean": .2, "update_norm_mean": .01}}
        for stage in ("kl", "nll") for epoch in range(3)]


def dataset(root):
    return unit(root, "dataset", train_count=4, validation_count=2, acquired_count=12, new_acquisitions=6,
        train_ids=[f"private-train-id-{i}" for i in range(4)], validation_ids=["v0", "v1"],
        coverage={"records": 6, "episodes": 3, "dominant_actions": {"0": 3, "1": 3},
            "return_bin_counts": {"75": 2, "80": 4}, "outcomes": {"success": 1, "collision": 1, "timeout": 1},
            "successes": 1, "degenerate_teacher_action_coverage": False,
            "controls": {"drive_mps_range": [0., .2], "steering_rad_range": [-.2, .2]},
            "episode_geometry": [{"episode_id": i, "initial_distance_m": .3 + i * .1,
                                  "initial_bearing_degrees": i * 20.} for i in range(3)]})


def bootstrap(root):
    return unit(root, "bootstrap", fisher_stability={
        "nested_draws": [{"draws_per_history": m, "history_count": 4, "trace": 2. + m,
                          "relative_l2": .4 / m, "absolute_l2": .02 / m} for m in (1, 2, 4)],
        "history_count": [{"history_count": n, "trace": 3. + n} for n in (2, 4)],
        "independent_subset": {"history_count": 2, "draws_per_history": 4, "relative_l2": .03, "absolute_l2": .01}},
        local_quadratic={"rows": [{"scale": s, "diagonal_quadratic": s*s,
            "exact_predictive_kl": 1.5*s*s, "corrected_observed_nll_change": .5*s*s} for s in (-.1, .1)]},
        normalization={"history_count": 4, "mean_trace": 6.},
        readiness={"ready": False, "fisher_numerically_ready": True, "fisher_stability_within_tolerance": False})


def panel(checkpoint="initial", *, difficulty="easy", n=0, seeds=(11, 12), successes=(True, False)):
    return {"checkpoint": checkpoint, "difficulty": difficulty, "acquired_count": n, "seeds": list(seeds),
            "rows": [{"seed": s, "success": success} for s, success in zip(seeds, successes)]}


def pilot(root, condition, *, seed=101, asset="common-anchor", seeds=(11, 12)):
    return unit(root, "pilot", name=condition, settings={"name": condition, "seed": seed, "initial_asset": asset},
        evaluations=[panel(c, difficulty=d, n=n, seeds=seeds) for c, n in (("initial", 0), ("final", 20))
                     for d in ("easy", "hard")])


def text_outputs(outputs):
    return "\n".join(o.get("text", "") + o.get("data", {}).get("text/plain", "") for o in outputs)


def rendered(book):
    return "\n".join(c.source + "\n" + text_outputs(c.get("outputs", [])) for c in book.cells)


def test_pending_report_is_read_only_concise_and_preserves_existing_code_order(tmp_path):
    root = tmp_path / "missing"
    book = nbformat.read(report.build_notebook(root, tmp_path / "report.ipynb"), as_version=4)
    nbformat.validate(book)
    assert not root.exists()
    assert "pending / not recorded" in rendered(book)
    assert "separate 10 h repair ceiling" in text_outputs(book.cells[1].outputs)
    code = [c for c in book.cells if c.cell_type == "code"]
    assert "Readiness measurements pending" in text_outputs(code[2].outputs)
    fix = [c for c in book.cells if c.metadata.get("plan19_phase3_fix1")]
    assert len(fix) == 2 and sum(len(c.source.split()) for c in fix) < 200
    assert all(c.outputs for c in code)


def test_committed_progress_before_first_fit_uses_pointer_not_mirror(tmp_path):
    live(tmp_path, {"status": "stationary collection", "repair_active_seconds": 360.,
        "cumulative_study_seconds": 38950., "remaining_budget_seconds": 35640.,
        "remaining_time_range_seconds": [7200., 9000.], "completed_trajectories": 0,
        "active_unit_progress": {"counts": {"train_histories": 256, "validation_histories": 64, "acquired_count": 400},
                                 "epoch": 2}})
    snapshot = shared.load_report(tmp_path)
    assert snapshot["complete_units"] == []
    row = report.progress_rows(snapshot)[0]
    assert row["Stage"] == "stationary collection"
    assert row["Usable train / validation"] == "256 / 64"
    assert row["Pilots complete / remaining"] == "0 / " + report.PENDING
    assert row["Remaining forecast / ceiling h"] == "2.00--2.50 / 9.900"
    assert "0.100 / 10.819" == row["Charged repair / study h"]
    assert row["Fisher"] == report.PENDING
    outputs = report.diagnostic_outputs(snapshot)
    assert not any("image/png" in o.get("data", {}) for o in outputs)
    assert "STALE" not in text_outputs(outputs)


def test_committed_diagnostic_epochs_plot_as_unsealed_work_not_readiness(tmp_path):
    path = live(tmp_path, {"status": "initialization: nll", "epoch": 2,
                          "diagnostic_curves": curves()}, name="units/fit-running")
    live(tmp_path, {"status": "fitting", "active_unit": str(path)})
    snapshot = shared.load_report(tmp_path)
    outputs = report.diagnostic_outputs(snapshot)
    assert len([o for o in outputs if "image/png" in o.get("data", {})]) == 1
    text = text_outputs(outputs)
    assert "Initialization in progress: committed diagnostic epochs; fit/selection not sealed" in text
    assert "do not establish readiness" in text
    assert snapshot["complete_units"] == []
    assert "readiness remains pending" in report.conclusion(snapshot)
    assert "kl: 2; nll: 2" in report.progress_rows(snapshot)[0]["Passes"]
    # Root mirrors can contain newer, uncommitted diagnostics without entering plots.
    write_json(path / "progress.json", {"schema_version": 1, "diagnostic_curves": [{"invented": True}]})
    assert report._live_fit(snapshot)[0]["summary"]["curves"] == curves()


def test_no_committed_epoch_stays_pending_and_unversioned_new_progress_is_rejected(tmp_path):
    live(tmp_path, {"diagnostic_curves": []}, name="units/fit-empty")
    snapshot = shared.load_report(tmp_path)
    assert not report._live_fit(snapshot)
    other = tmp_path / "unversioned"
    path = live(other, {"status": "fitting"}, schema=False)
    with pytest.raises(shared.ArtifactError, match="progress schema"):
        report._committed_progress(path)
    assert report._committed_progress(path, strict=False)["status"] == "fitting"


def test_small_rung_panels_plot_vs_data_size_without_pooling_selected_full_panels(tmp_path, monkeypatch):
    for target in (256, 512):
        for checkpoint in ("distillation", "accepted_nll"):
            unit(tmp_path, "evaluation", name=f"{target}-{checkpoint}", checkpoint=checkpoint,
                 train_count=target, small_panel_not_confirmation=True,
                 evaluations=[panel(seeds=(11, 12, 13, 14, 15), successes=(True, False, False, False, False))])
    unit(tmp_path, "evaluation", name="full-selected", checkpoint="accepted_nll",
         evaluations=[panel(successes=(False, False))])
    observed = []
    original = report._figure

    def capture(figure, description):
        observed.extend((list(line.get_xdata()), list(line.get_ydata())) for line in figure.axes[0].lines)
        return original(figure, description)

    monkeypatch.setattr(report, "_figure", capture)
    outputs = report.diagnostic_outputs(shared.load_report(tmp_path))
    assert observed == [([256, 512], [.2, .2]), ([256, 512], [.2, .2])]
    text = text_outputs(outputs)
    assert "20-percentage-point resolution" in text and "0 / 2" in text


def test_provenance_integrity_and_software_units_do_not_invent_numerical_results(tmp_path):
    for kind in ("provenance", "integrity", "software"):
        unit(tmp_path, kind, passed=True)
    snapshot = shared.load_report(tmp_path)
    assert len(report._units(snapshot)) == 3
    assert "pending" in report.progress_rows(snapshot)[0]["Fisher"]
    assert "readiness remains pending" in report.conclusion(snapshot)
    report.build_notebook(tmp_path, tmp_path / "out.ipynb")


@pytest.mark.parametrize("kind,fields,match", [
    ("fit", {"curves": [], "readiness": {"ready": False}}, "nonempty fixed-set"),
    ("fit", {"curves": [{}], "readiness": {"ready": False}}, "stage and epoch"),
    ("fit", {"curves": curves(), "readiness": {}}, "nonempty readiness"),
    ("fit", {"curves": curves(), "readiness": {"ready": "false"}}, "boolean"),
    ("bootstrap", {}, "Fisher diagnostics"),
    ("evaluation", {"checkpoint": "accepted_nll", "evaluations": []}, "evaluation panels"),
    ("new_unknown", {}, "unknown initialization"),
])
def test_completed_but_incompatible_summaries_fail_clearly(tmp_path, kind, fields, match):
    unit(tmp_path, kind, **fields)
    with pytest.raises(shared.ArtifactError, match="schema-incompatible.*" + match):
        report.diagnostic_outputs(shared.load_report(tmp_path))


def test_tampered_scalar_or_live_transaction_is_rejected_without_repair(tmp_path):
    path = unit(tmp_path, "fit", curves=curves(), readiness={"ready": False})
    (path / "summary.json").write_text((path / "summary.json").read_text() + " ")
    with pytest.raises(shared.ArtifactError, match="hash/size"):
        shared.load_report(tmp_path)
    assert (path / "COMPLETED").is_file()
    other = tmp_path / "live-only"
    path = live(other, {"status": "fitting"})
    (path / "transactions/epoch-2/progress.json").write_text("{}")
    with pytest.raises(shared.ArtifactError, match="hash/size"):
        report._committed_progress(path)


def test_nested_numerical_schema_yields_four_compact_diagnostic_groups(tmp_path, monkeypatch):
    dataset(tmp_path)
    unit(tmp_path, "fit", curves=curves(), readiness={"ready": False, "optimization_ready": False},
         episodes=[{"private_payload": "not for notebook"}] * 1000)
    bootstrap(tmp_path)
    unit(tmp_path, "evaluation", checkpoint="accepted_nll", evaluations=[panel()])
    pilot(tmp_path, "replay_ewc")
    pilot(tmp_path, "stm_fd")
    captured = []
    original = report._figure

    def capture(figure, description):
        captured.append([[list(line.get_ydata()) for line in ax.lines] for ax in figure.axes])
        return original(figure, description)

    monkeypatch.setattr(report, "_figure", capture)
    outputs = report.diagnostic_outputs(shared.load_report(tmp_path))
    images = [o for o in outputs if "image/png" in o.get("data", {})]
    assert len(images) == 4
    assert all(base64.b64decode(o.data["image/png"]).startswith(b"\x89PNG") for o in images)
    assert [5., 4., 3.] in captured[1][0]
    assert [.8, .7, .6] == pytest.approx(captured[1][1][0])
    assert [3., 4., 6.] in captured[2][0]
    assert captured[2][4] and captured[2][5]
    assert len(captured[3][1]) == len(captured[3][2]) == 2
    text = text_outputs(outputs)
    assert "private-train-id" not in text and "private_payload" not in text
    assert len(text) < 7000 and len(outputs) < 20
    assert "first-order" in text and "not proof of LAN" in text


@pytest.mark.parametrize("difference", ["seed", "asset", "panels"])
def test_unpaired_pilots_never_become_a_paired_comparison(tmp_path, monkeypatch, difference):
    pilot(tmp_path, "replay_ewc")
    pilot(tmp_path, "stm_fd", seed=102 if difference == "seed" else 101,
          asset="different" if difference == "asset" else "common-anchor",
          seeds=(21, 22) if difference == "panels" else (11, 12))
    lines = []
    original = report._figure

    def capture(figure, description):
        lines.extend(len(ax.lines) for ax in figure.axes[1:])
        return original(figure, description)

    monkeypatch.setattr(report, "_figure", capture)
    report.diagnostic_outputs(shared.load_report(tmp_path))
    assert lines == [0, 0]


def test_invalid_or_incomplete_boolean_panels_remain_pending():
    assert report._panel(panel(successes=(True, None)))[0] is None
    assert report._panel(panel(seeds=(11, 11)))[0] is None
    value = panel()
    value["seeds"].append(13)
    assert report._panel(value)[0] is None


@pytest.mark.parametrize("ready", [True, False])
def test_conclusion_separates_fit_readiness_and_fd_evidence(tmp_path, ready):
    unit(tmp_path, "fit", curves=curves(), readiness={"ready": ready})
    unit(tmp_path, "campaign", status="ready_for_checkin", initialization_ready=ready, decision={"retention_matched": False})
    text = report.conclusion(shared.load_report(tmp_path))
    assert ("declared readiness rule passed" if ready else "not a negative FD") in text
    assert "not retention matched" in text and "No sample-savings claim" in text
    assert "no automatic Phase 4 or physical motion" in text


def test_historical_phase3_filter_keeps_old_results_without_duplicate_repair_campaign(tmp_path):
    old = tmp_path / "phase3/development-v1/orchestration"
    old.mkdir(parents=True)
    write_json(old / "progress.json", {"phase": 3, "status": "old mirror"})
    new = live(tmp_path, {"status": "new collection"})
    value = {"root": str(tmp_path), "complete_units": [], "live": [
        {"path": str(old), "phase": 3, "status": "historical committed", "counts": {}},
        {"path": str(new), "phase": 3, "status": "new", "counts": {}}]}
    from numerical_experiment.picar.phase3_report import progress_rows
    rows = progress_rows(report.historical_phase3_snapshot(value))
    assert len(rows) == 1 and rows[0]["Campaign"] == "development-v1"
    assert rows[0]["Status"] == "historical committed"


def test_append_is_chronological_idempotent_and_retains_historical_closeout(tmp_path):
    unit(tmp_path, "software", passed=True)
    snapshot = shared.load_report(tmp_path)
    old = nbformat.v4.new_markdown_cell("### Phase 3: original results", metadata={"plan19_phase3": True})
    book = shared.make_notebook(tmp_path)
    book.cells.insert(-1, old)
    report.append_fix1(book, snapshot, historical_conclusion="## Conclusion\n\nOriginal result remains unresolved.")
    report.append_fix1(book, snapshot, historical_conclusion="## Conclusion\n\nOriginal result remains unresolved.")
    nbformat.validate(book)
    sources = [c.source for c in book.cells]
    index = sources.index(old.source)
    assert "Historical Phase 3 Closeout" in sources[index + 1]
    assert "Original result remains unresolved" in sources[index + 1]
    assert sources[index + 2].startswith("### Phase 3 Fix 1:")
    assert sources[-1].startswith("## Conclusion")
    assert sum(s.startswith("### Phase 3 Fix 1:") for s in sources) == 1
    assert book.cells[1].source.count("display_progress(snapshot)") == 1


@pytest.mark.parametrize("partial", [False, True])
def test_all_completion_cells_execute_with_torch_and_runtime_imports_blocked(tmp_path, partial):
    if partial:
        path = live(tmp_path, {"status": "initialization: nll", "epoch": 2,
                              "diagnostic_curves": curves()}, name="units/fit-running")
        live(tmp_path, {"status": "fitting", "active_unit": str(path)})
    else:
        dataset(tmp_path)
        unit(tmp_path, "fit", curves=curves(), readiness={"ready": False})
        bootstrap(tmp_path)
        unit(tmp_path, "evaluation", checkpoint="accepted_nll", evaluations=[panel()])
    script = """
import builtins, contextlib, io, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    prefixes = ('torch', 'transformers', 'peft', 'pybullet', 'bitsandbytes',
                'numerical_experiment.picar.models', 'numerical_experiment.picar.storage',
                'numerical_experiment.picar.learner', 'numerical_experiment.picar_initialization.numerics',
                'numerical_experiment.picar_initialization.campaign', 'numerical_experiment.picar_initialization.config',
                'numerical_experiment.picar_initialization.data', 'numerical_experiment.picar_phase3')
    if any(name == p or name.startswith(p + '.') for p in prefixes):
        raise AssertionError('runtime import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import nbformat
from numerical_experiment.picar_initialization.report import build_notebook
path = build_notebook(sys.argv[1], sys.argv[2])
book = nbformat.read(path, as_version=4)
namespace = {}
with contextlib.redirect_stdout(io.StringIO()):
    for cell in book.cells:
        if cell.cell_type == 'code':
            exec(compile(cell.source, '<report>', 'exec'), namespace)
assert 'torch' not in sys.modules
assert all(c.outputs for c in book.cells if c.cell_type == 'code')
nbformat.validate(book)
print('all artifact-only cells executed')
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(tmp_path / "report.ipynb")],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2], timeout=60,
        env={**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")})
    assert result.returncode == 0, result.stderr
    assert "all artifact-only cells executed" in result.stdout
def test_dense_return_bins_use_numeric_axis_without_a_label_per_bin(monkeypatch):
    from numerical_experiment.picar_initialization import report as repair
    monkeypatch.setattr(repair, "_figure", lambda figure, description: figure)
    figure = repair._coverage([{"summary": {"kind": "initialization_dataset",
        "train_count": 259, "validation_count": 65,
        "coverage": {"return_bin_counts": {str(i): 1 for i in range(161)}}}}])
    axis = figure.axes[2]
    assert len(axis.get_xticks()) <= 9
    assert max(bar.get_x() for bar in axis.patches) > 150
    figure.clear()


def test_paired_curves_preserve_different_actual_episode_boundary_counts(monkeypatch):
    monkeypatch.setattr(report, "_figure", lambda figure, description: figure)
    units = []
    for condition, endpoint in (("replay_ewc", 634), ("stm_fd", 628)):
        units.append({"summary": {"kind": "initialization_pilot",
            "settings": {"name": condition, "seed": 195101, "initial_asset": "same"},
            "evaluations": [panel(c, difficulty=d, n=n)
                for c, n in (("initial", 0), ("midpoint", endpoint // 2), ("final", endpoint))
                for d in ("easy", "hard")]}})
    figure = report._capability(units)
    assert [list(line.get_xdata()) for line in figure.axes[1].lines] == [[0, 317, 634], [0, 314, 628]]
    assert len(figure.axes[2].lines) == 2
    figure.clear()


def test_independent_reference_uses_its_own_version_not_the_campaign_version():
    value = {"path": "reference", "config": {"version": report.REFERENCE_VERSION},
        "summary": {"kind": "initialization_mc_reference", "primary_unchanged": True,
            "reference_is_oracle": False, "comparisons": {"primary_4_vs_reference": {"relative_l2": .2}}}}
    assert report._validate(value) == "mc_reference"
    value["config"]["version"] = report.VERSION
    with pytest.raises(shared.ArtifactError, match="unexpected initialization version"):
        report._validate(value)


def test_completed_pilot_table_preserves_counts_and_does_not_invent_pending_outcomes(tmp_path):
    pilot(tmp_path, "replay_ewc")
    values = report._units(shared.load_report(tmp_path))
    row = report._pilot_rows(values)[0]
    assert row["Method"] == "Replay/EWC"
    assert row["A successes: initial -> final"] == "1/2 -> 1/2"
    assert row["B successes: initial -> final"] == "1/2 -> 1/2"
    assert row["Active min"] is None
    assert len(report._pilot_rows(values)) == 1
    values[0]["summary"]["evaluations"] = [panel()]
    assert report._pilot_rows(values)[0]["A successes: initial -> final"] == report.PENDING
    assert report._pilot_rows(values)[0]["B successes: initial -> final"] == report.PENDING


def test_conclusion_states_zero_acquisition_without_claiming_fd_failure(tmp_path):
    pilot(tmp_path, "replay_ewc")
    snapshot = shared.load_report(tmp_path)
    for p in snapshot["complete_units"][0]["summary"]["evaluations"]:
        if p["difficulty"] == "hard":
            for row in p["rows"]:
                row["success"] = False
    text = report.conclusion(snapshot)
    assert "Every completed pilot ended at zero hard-B successes" in text
    assert "not established" in text and "No sample-savings claim" in text


def test_final_conclusion_reports_retention_counts_and_qualified_next_action(tmp_path):
    pilot(tmp_path, "replay_ewc")
    pilot(tmp_path, "stm_fd")
    unit(tmp_path, "campaign", status="ready_for_checkin", decision={"control_learning_observed": False})
    text = report.conclusion(shared.load_report(tmp_path))
    assert "Replay/EWC 1/2; STM FD 1/2" in text
    assert "one result per completed development trajectory" in text
    assert "separately versioned diagnostic of steering coverage" in text
    assert "do not isolate the acquisition bottleneck" in text

