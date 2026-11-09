"""Phase 7 scalar provenance, paired inference, and immutable chronology."""

from copy import deepcopy
import fcntl
import json
from pathlib import Path

import nbformat
import numpy as np
import pytest

from mnist_experiment.combined_memory import analysis, report as original
from mnist_experiment.combined_memory.broad_drift import report as broad
from mnist_experiment.combined_memory.retention_match import report as phase6
from mnist_experiment.combined_memory.low_ridge import report, runtime


def config(**changes):
    return dict(dict(protocol=report.PROTOCOL, smoke=True, analysis_novel_count=4,
        samples_per_step=2, hold_steps=1, ramp_steps=2, stream_size=12,
        bootstrap_draws=32, seed=18110907, main_replicas=8, stationary_replicas=2,
        retention_tolerance=.1, fd_strength=.5691316767656495), **changes)


def recipe(arm):
    return dict(arm=arm, ridge=dict(A=2., B=1., C=.5, D=.1, E=.1)[arm],
                strength=0. if arm == "A" else .5691316767656495,
                process_scale=1e-5 if arm in "DE" else 1e-4)


def selection():
    return dict(frozen=True, calibration_status="passed", recipes={arm: recipe(arm) for arm in analysis.ARMS},
        main_replicas=8, stationary_replicas=2, bootstrap_seed=196,
        calibration={}, matching=dict(passed=True, ranges=dict(Havg=.04, H100=.04), means={}),
        members=[0, 1, 2, 3], reference_arm="B", reference_ridge=1., efficiency_reference_arm="A")


def record(arm="A", replica=0, cohort="fresh", cfg=None):
    cfg = cfg or config()
    m, hold, ramp = (cfg[k] for k in ("samples_per_step", "hold_steps", "ramp_steps"))
    total = 2 * (hold + ramp)
    endpoint = dict(A=8., B=6., C=5., D=4., E=3.)[arm]
    rows = []
    for step in range(total + 1):
        n = 0 if cohort == "stationary" else m * max(0, min(step - hold, ramp))
        risk = 20. if step == 0 else 16. - (16. - endpoint) * n / cfg["analysis_novel_count"]
        old = 4. if step == 0 else (2.02 + .01 * analysis.ARMS.index(arm) + .003 * replica) ** 2
        if step > hold + ramp:
            risk, old = 999., 900.
        rows.append(dict(accepted_step=step, incoming_grids=m * step, outbound_count=n,
            novel_count=n, kind="initial" if step == 0 else "regular", novel_mse=risk, old_mse=old,
            current_mse=1. + step / total, old_accuracy=.75, novel_accuracy=.5, current_accuracy=.7,
            frozen_current_mse=3., learning_seconds=step * (2. if arm == "E" else 1.)))
    return dict(unit_id=f"{cohort}-{replica}-{arm}", status="completed", metrics=rows,
        summary=analysis.summarize_path(rows, cfg), diagnostics=[],
        metadata=dict(kind="trajectory", cohort=cohort, replica=replica, protocol=report.PROTOCOL,
            recipe=recipe(arm), pairing_asset_hash=f"{cohort}-assets-{replica}",
            global_sign=-1 if replica % 2 else 1,
            imported=cohort == "development", origin_root="historical" if cohort == "development" else None))


def records(cohort="fresh", cfg=None, replicas=range(8)):
    return [record(arm, replica, cohort, cfg) for replica in replicas for arm in analysis.ARMS]


def snapshot(rows=None, cfg=None, **changes):
    return dict(dict(records=records() if rows is None else rows, cfg=cfg or config(),
        selection=selection(), status=dict(calibration_status="passed"), development={},
        inputs=dict(scalar_exports={})), **changes)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(original._bytes(value))


def identity(root, cfg):
    scientific = dict(protocol=report.PROTOCOL, config=cfg, metric_schema_version=1, artifact_schema_version=1)
    write_json(root / "identity.json", dict(schema=1, identity=scientific,
        identity_hash=original._hash(original._bytes(scientific))))


def exported(root, generation, name="analysis.json"):
    return json.loads((root / "report" / generation / name).read_bytes())


@pytest.fixture
def scalar_reader(monkeypatch):
    state = dict(records=[], provenance=dict(files={}, imports=[], identity_hash="scalar-fixture"))

    def read(root, check=lambda: None):
        check()
        return deepcopy(state["records"]), deepcopy(state["provenance"])

    def decision(root, uid, name, check=lambda: None):
        check()
        assert (uid, name) == ("decision-selection", "decision.json")
        value = state.get("decision") or json.loads((root / "selection.json").read_bytes())
        return value, original._hash(original._bytes(value))

    monkeypatch.setattr(runtime, "read_exported_records", read)
    monkeypatch.setattr(runtime, "checked_scalar", decision)
    return state


def reseal_history(root):
    path = root / "report/sealed-phase6"
    digest = "historical-phase6-digest"
    write_json(path / "COMPLETED", dict(schema=1, source_digest=digest,
        files={name: original._hash((path / name).read_bytes()) for name in ("results.ipynb", "generation.json")}))
    write_json(root / "report/latest.json", dict(schema=1, generation=path.name, source_digest=digest))


@pytest.fixture
def history(tmp_path, monkeypatch):
    root = tmp_path / "retention_match_v1"
    book = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Original Phase 0--5\n\nPreserved numerical qualifications."),
        nbformat.v4.new_code_cell("1 + 1", execution_count=1,
            outputs=[nbformat.v4.new_output("execute_result", data={"text/plain": "2"}, execution_count=1)]),
        nbformat.v4.new_markdown_cell("## Phase 6: Retention-Matched Recomparison\n\nValid comparison."),
        nbformat.v4.new_markdown_cell("### Phase 6 Conclusion\n\nCalibration passed. Earlier sustained quality, "
            "not twice-as-fast initial learning. Intervals remain essential.")],
        metadata=dict(custom="preserve", plan18_phase6_report=dict(protocol=phase6.PROTOCOL)))
    path = root / "report/sealed-phase6"
    path.mkdir(parents=True)
    (path / "results.ipynb").write_bytes(nbformat.writes(book).encode())
    write_json(path / "generation.json", dict(schema=1, protocol=phase6.PROTOCOL, status="completed"))
    reseal_history(root)
    monkeypatch.setattr(report, "HISTORY_ROOT", root)
    return root, book


@pytest.fixture
def fast_figures(monkeypatch):
    def render(snapshot, result):
        figure, axes = original._figure(1, 1)
        axes[0].plot([0, 1], [0, 1])
        return figure

    for name in ("_development", "_prediction", "_ridge_motion", "_bootstrap_histogram"):
        monkeypatch.setattr(report, name, render)
    monkeypatch.setattr(broad, "_diagnostics", render)


def test_real_report_preserves_history_and_never_opens_checkpoints(tmp_path, history, scalar_reader, monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        pytest.fail("report attempted model/checkpoint or training access")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(runtime, "load_readonly_checkpoint", forbidden)
    opened = Path.open

    def scalar_only(path, *args, **kwargs):
        assert path.suffix not in (".pt", ".pth", ".npz"), "non-scalar artifact opened"
        return opened(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", scalar_only)
    root, public = tmp_path / "low_ridge", tmp_path / "public.ipynb"
    public.write_text("public is not history")
    cfg = config()
    identity(root, cfg)
    scalar_reader["records"] = records("development", replicas=range(4)) + records() + records("stationary", replicas=range(2))
    write_json(root / "selection.json", selection())
    write_json(root / "development.json", dict(members=list(range(4))))
    write_json(root / "status.json", dict(protocol=report.PROTOCOL, phase="7E", status="completed",
        calibration_status="passed", active_seconds=21*3600, additional_active_seconds=2*3600,
        baseline_seconds=19*3600, incremental_allowance_seconds=72000, forecast_remaining_seconds=0))
    write_json(root / "calibration_completed.json", dict(status="passed"))
    generation = report.build_report(root, notebook_path=public)
    book = nbformat.read(root / "smoke_results.ipynb", as_version=4)
    nbformat.validate(book)
    assert book.cells[1:1 + len(history[1].cells)] == history[1].cells
    assert book.metadata.custom == "preserve"
    assert book.metadata.plan18_phase7_report.protocol == report.PROTOCOL
    cells = book.cells[1 + len(history[1].cells):]
    assert len(cells) == 11 and sum(bool(c.get("attachments")) for c in cells) == 5
    assert all(c.cell_type == "markdown" for c in cells)
    text = "\n".join(c.source for c in cells)
    for required in ("Pi, Gain And Motion Diagnostics", "not confidence intervals", "H4", "ridge 1",
                     "calibrated A", "mean trajectories", "bootstrap median", "Stationary paths have no acquisition clock or S"):
        assert required in text
    assert "H100" not in text
    assert "imported 20, new 0" in book.cells[0].source
    assert "fresh 8/8 paired blocks" in book.cells[0].source
    assert "2 / 20 h" in book.cells[0].source and "family-cumulative: 21 h" in book.cells[0].source
    assert public.read_text() == "public is not history"
    result = exported(root, generation["generation"])
    assert result["status"] == "completed" and result["protocol"] == report.PROTOCOL
    assert len(result["retention_equivalence"]) == 10
    assert result["cohorts"]["stationary"]["efficiency"] == {}
    assert generation["figure_count"] == 5


def test_reference_roles_and_whole_pair_retention_draws_are_unchanged():
    source = snapshot()
    result = report._summarize(source)
    inherited = phase6._summarize(source)
    assert result["calibration"]["passed"]
    assert result["reference_roles"] == dict(development_retention="B", primary_sample_equivalence="A", secondary="E-B")
    assert result["efficiency"] == inherited["efficiency"]
    assert result["efficiency"]["B-A"]["point"]["target_mse"] == 8.
    assert result["efficiency"]["E-B"]["point"]["target_mse"] == 6.
    assert result["efficiency"]["A-A"]["point"]["saved_fraction"] == 0
    assert result["retention_equivalence"] == inherited["retention_equivalence"]
    draws = analysis.paired_draws(list(range(8)), 32, 196)
    for name, item in result["retention_equivalence"].items():
        left, right = name.split("-")
        for field in ("Havg", "H100"):
            difference = np.asarray(result["methods"][left][field]["per_replica"]) - np.asarray(result["methods"][right][field]["per_replica"])
            assert item[field] == dict(analysis._interval(difference, draws), equivalent=True)


@pytest.mark.parametrize("change", ["old_B", "retention_role", "efficiency_role", "unmatched", "members"])
def test_old_tuple_or_wrong_reference_roles_cannot_authorize_fresh_inference(change):
    source = snapshot()
    if change == "old_B":
        source["selection"]["recipes"]["B"]["ridge"] = 10.
        for row in source["records"]:
            if row["metadata"]["recipe"]["arm"] == "B":
                row["metadata"]["recipe"]["ridge"] = 10.
    elif change == "retention_role":
        source["selection"]["reference_arm"] = "A"
    elif change == "efficiency_role":
        source["selection"]["efficiency_reference_arm"] = "B"
    elif change == "unmatched":
        source["selection"]["matching"]["ranges"]["Havg"] = .10001
    else:
        source["selection"]["members"] = [0, 1]
    with pytest.raises(ValueError, match="calibration"):
        report._summarize(source)


def test_matching_tolerance_uses_the_original_epsilon():
    source = snapshot()
    source["selection"]["matching"]["ranges"] = dict(Havg=.1 + 1e-9, H100=.1 + 1e-9)
    assert report._summarize(source)["calibration"]["passed"]


@pytest.mark.parametrize("replicas", [(0,), (0, 1), (0, 1, 2), (0, 2, 4, 6)])
def test_intervals_and_histograms_wait_for_two_per_actual_sign(replicas):
    source = snapshot(records(replicas=replicas))
    result = report._summarize(source)
    assert result["interval_status"] == "descriptive_insufficient_sign_replicates"
    assert result["cohorts"]["fresh"]["bootstrap"]["draws"] == 0
    assert result["bootstrap_saved_fractions"] == {}
    assert all(not item["equivalent"] for item in result["retention_equivalence"].values())
    for item in result["efficiency"].values():
        assert item["ci"] is None and not item["sample_saving_supported"]
        assert not item["retention_supported"]
    assert all(value["ci"] is None for item in result["comparisons"].values() for value in item.values())


def test_scalar_diagnostics_plot_actual_saved_values_and_fresh_pair_means():
    rows = records("development", replicas=range(4)) + records() + records("stationary", replicas=range(2))
    for row in rows:
        replica = row["metadata"]["replica"]
        value = replica + 1 if row["metadata"]["cohort"] == "fresh" else 1000.
        row["diagnostics"] = [dict(incoming_grids=t, s=2*value, kappa=3*value, applied_pi=.1,
            raw_pi=.2, ltm_move=4*value, stm_move=5*value, noise_scale=6*value,
            motion_energy=value, uncertainty_energy=1., kalman_level_gain_mean=.01*value,
            kalman_level_gain_p10=.005*value, kalman_level_gain_p90=.015*value,
            kalman_velocity_gain_mean=.001*value, correction_norm=value, projection_norm=0.) for t in (2, 4)]
    # Save values that intentionally differ from c*s: never silently reconstruct kappa.
    source = snapshot(rows)
    result = report._summarize(source)
    scalars = result["scalar_diagnostics"]
    assert scalars["cohort"] == "fresh"
    assert scalars["methods"]["B"]["s"]["pooled_mean"] == 9.
    assert scalars["methods"]["B"]["kappa"]["pooled_mean"] == 13.5
    figure = report._ridge_motion(source, result)
    try:
        for axis, label, expected in ((0, "B", 1.), (1, "B", 9.), (2, "B", 13.5),
                                      (3, "D", 9.), (4, "E", 121.5), (5, "B LTM", 18.), (5, "B STM", 22.5)):
            line = next(line for line in figure.axes[axis].lines if line.get_label() == label)
            np.testing.assert_allclose(line.get_xdata(), [2, 4])
            np.testing.assert_allclose(line.get_ydata(), [expected, expected])
    finally:
        figure.clear()
    figure = broad._diagnostics(source, result)
    try:
        for axis, label, expected in ((0, "D applied pi", .1), (1, "E level", .045),
                                      (3, "D one-grid V", 27.), (3, "D batch V/m", 13.5),
                                      (3, "E motion + uncertainty", 5.5)):
            line = next(line for line in figure.axes[axis].lines if line.get_label() == label)
            np.testing.assert_allclose(line.get_ydata(), [expected, expected])
    finally:
        figure.clear()


def varied_snapshot():
    rows = records(replicas=range(4))
    for row in rows:
        arm, replica = row["metadata"]["recipe"]["arm"], row["metadata"]["replica"]
        # Irregular signs exercise recorded strata, not an accidental parity split.
        row["metadata"]["global_sign"] = -1 if replica < 2 else 1
        end = dict(A=8., B=6., C=5., D=4., E=3.)[arm] + .4 * replica
        for metric in row["metrics"]:
            if 1 <= metric["accepted_step"] <= 3:
                metric["novel_mse"] = 16. - (16. - end) * metric["novel_count"] / 4
            metric["learning_seconds"] *= 1. + .2 * replica
        row["summary"] = analysis.summarize_path(row["metrics"], config())
    return snapshot(rows)


def test_histograms_recompute_targets_and_entire_sign_stratified_pairs_from_saved_curves():
    source = varied_snapshot()
    result = report._summarize(source)
    replicas = result["cohorts"]["fresh"]["replicas"]
    draws = analysis.paired_draws(replicas, 32, 196, [-1, -1, 1, 1])
    arrays = {}
    for arm in ("A", "B", "E"):
        group = [next(r for r in source["records"] if r["metadata"]["recipe"]["arm"] == arm and r["metadata"]["replica"] == replica)
                 for replica in replicas]
        curves = [analysis.novel_curve(r["metrics"], source["cfg"]) for r in group]
        arrays[arm] = {k: np.asarray([c[k] for c in curves]) for k in
                       ("novel_mse", "old_mse", "learning_seconds", "incoming_grids", "frozen_novel_mse")}
    for arm in ("B", "E"):
        expected = [analysis._equivalence(arrays, arm, "A", sample) for sample in draws]
        item = result["bootstrap_saved_fractions"][f"{arm}-A"]
        assert item["status"] == "estimable" and item["draws"] == 32
        assert item["signs"] == [-1, -1, 1, 1]
        assert item["target_mse"] == [r["target_mse"] for r in expected]
        assert len(set(item["target_mse"])) > 1
        for field in ("saved_fraction", "n"):
            np.testing.assert_allclose(item[field], [r[field] for r in expected])
        np.testing.assert_allclose(np.quantile(item["saved_fraction"], [.025, .975]),
                                   result["efficiency"][f"{arm}-A"]["ci"]["saved_fraction"])
        assert item["point_from_mean_curves"] == result["efficiency"][f"{arm}-A"]["point"]["saved_fraction"]


def test_failed_bootstrap_crossings_are_retained_not_zeroed_or_conditioned_away():
    source = varied_snapshot()
    for row in source["records"]:
        if row["metadata"]["recipe"]["arm"] == "B":
            end = 14. if row["metadata"]["replica"] == 0 else 6.
            for metric in row["metrics"]:
                if 1 <= metric["accepted_step"] <= 3:
                    metric["novel_mse"] = 16. - (16. - end) * metric["novel_count"] / 4
            row["summary"] = analysis.summarize_path(row["metrics"], source["cfg"])
    result = report._summarize(source)
    item = result["bootstrap_saved_fractions"]["B-A"]
    assert item["status"] == "withheld_unsupported_crossings"
    assert len(item["saved_fraction"]) == 32 and None in item["saved_fraction"]
    assert 0 < item["failure_frequency"] < 1
    assert item["failure_frequency"] == result["efficiency"]["B-A"]["bootstrap_failure_frequency"]
    assert result["efficiency"]["B-A"]["ci"] is None
    figure = report._bootstrap_histogram(source, result)
    try:
        assert not figure.axes[0].patches
        assert any("unsupported crossings" in text.get_text() for text in figure.axes[0].texts)
    finally:
        figure.clear()


def test_crossing_diagnostic_preserves_first_versus_persistent_crossing():
    cfg = config(analysis_novel_count=8, samples_per_step=2, ramp_steps=4, stream_size=20)
    source = snapshot(records(cfg=cfg, replicas=range(4)), cfg)
    for row in source["records"]:
        arm = row["metadata"]["recipe"]["arm"]
        risks = [18., 6., 12., 7.] if arm == "B" else [18., 12., 14., 8.]
        for metric in row["metrics"]:
            if 2 <= metric["accepted_step"] <= 5:
                metric["novel_mse"] = risks[metric["accepted_step"] - 2]
        row["summary"] = analysis.summarize_path(row["metrics"], cfg)
    result = report._summarize(source)
    clock = result["crossing_diagnostics"]["B-A"]
    assert clock["first_crossing"] < clock["persistent_crossing"]
    assert clock["target_mse"] == 8.
    assert "Persistent last crossing" in report._efficiency(result)
    assert f"{clock['persistent_crossing']:.1f}" in report._efficiency(result)


@pytest.mark.parametrize("reference_crossing", [2.25, 4.])
@pytest.mark.parametrize("risk_bounds", [[-.02, .01], [-.03, -.01], None])
def test_conclusion_distinguishes_endpoint_budget_savings_from_reference_timing(reference_crossing, risk_bounds):
    source = snapshot()
    result = report._summarize(source)
    result["crossing_diagnostics"]["A-A"]["persistent_crossing"] = reference_crossing
    result["comparisons"]["B-A"]["novel_auc"]["ci"] = risk_bounds
    before = deepcopy(result)
    text = report._conclusion(source, result)
    point = result["efficiency"]["B-A"]["point"]
    assert f"{point['n']:.1f} of 4 allocated grids" in text
    assert f"{point['saved_percent']:.3f}% endpoint-budget saving" in text
    assert f"A's own sustained crossing is {reference_crossing:.1f} grids" in text
    assert "not additional timing advantage" in text
    assert "do not establish two-sided equivalence" in text
    assert ("acquisition-risk interval includes zero" in text) == (risk_bounds == [-.02, .01])
    assert result == before


def test_immutable_rebuild_restores_public_output_and_source_history_seals_change_generation(tmp_path, history, scalar_reader, fast_figures, monkeypatch):
    root, public = tmp_path / "production", tmp_path / "public.ipynb"
    cfg = config(smoke=False, analysis_novel_count=1216, samples_per_step=4,
                 hold_steps=160, ramp_steps=304, stream_size=3712)
    identity(root, cfg)
    before = {p: p.read_bytes() for p in history[0].rglob("*") if p.is_file()}
    first = report.build_report(root, notebook_path=public)
    data = public.read_bytes()
    assert not report.build_report(root, notebook_path=public)["changed"]
    public.write_text("unsealed edits are not history")
    assert not report.build_report(root, notebook_path=public)["changed"]
    assert public.read_bytes() == data
    forced = report.build_report(root, force=True, notebook_path=public)
    assert forced["generation"] != first["generation"]
    assert report.build_report(root, notebook_path=public)["generation"] == forced["generation"]
    write_json(root / "status.json", dict(protocol=report.PROTOCOL, phase="7C", calibration_status="pending"))
    updated = report.build_report(root, notebook_path=public)
    assert updated["generation"] != first["generation"]
    assert (root / "report" / first["generation"] / "results.ipynb").read_bytes() == data
    assert all(p.read_bytes() == value for p, value in before.items())
    book = nbformat.read(public, as_version=4)
    assert book.cells[1:1 + len(history[1].cells)] == history[1].cells
    assert sum(c.source.startswith("## Phase 7:") for c in book.cells) == 1
    sources = dict(report._SOURCE_BYTES)
    sources["report_source.py"] += b"\n# imported implementation changed\n"
    monkeypatch.setattr(report, "_SOURCE_BYTES", sources)
    reimplemented = report.build_report(root, notebook_path=public)
    assert reimplemented["generation"] != updated["generation"]
    inputs = exported(root, reimplemented["generation"], "inputs.json")
    for name, content in sources.items():
        assert (root / "report" / reimplemented["generation"] / name).read_bytes() == content
        assert inputs["implementation"][name] == original._hash(content)
    scalar_reader["provenance"]["imports"] = [dict(unit="import", sha256="new-seal")]
    assert report.build_report(root, notebook_path=public)["generation"] != reimplemented["generation"]
    path = history[0] / "report/sealed-phase6/results.ipynb"
    old = nbformat.read(path, as_version=4)
    old.cells[-1].source += "\nRetained extra qualification."
    nbformat.write(old, path)
    reseal_history(history[0])
    assert report.build_report(root, notebook_path=public)["generation"] != reimplemented["generation"]


@pytest.mark.parametrize("defect", ["missing", "protocol", "status", "tampered", "symlink", "phase7"])
def test_production_requires_verified_complete_phase6_history(tmp_path, history, scalar_reader, fast_figures, defect):
    root = history[0]
    if defect == "missing":
        root = tmp_path / "missing"
    elif defect in ("protocol", "status"):
        write_json(root / "report/sealed-phase6/generation.json", dict(schema=1,
            protocol=broad.PROTOCOL if defect == "protocol" else phase6.PROTOCOL,
            status="pending" if defect == "status" else "completed"))
        reseal_history(root)
    elif defect == "symlink":
        path = root / "report/latest.json"
        saved = path.read_bytes()
        path.unlink()
        write_json(root / "elsewhere.json", json.loads(saved))
        path.symlink_to(root / "elsewhere.json")
    else:
        path = root / "report/sealed-phase6/results.ipynb"
        book = nbformat.read(path, as_version=4)
        book.cells.append(nbformat.v4.new_markdown_cell("## Phase 7: untrusted append"))
        nbformat.write(book, path)
        if defect == "phase7":
            reseal_history(root)
    with pytest.raises(ValueError, match="history|Phase 6|generation changed"):
        report.build_report(tmp_path / "production", config(smoke=False, analysis_novel_count=1216),
                            notebook_path=tmp_path / "public.ipynb", history_root=root)


def test_source_archive_is_imported_bytes_not_later_disk_edits(tmp_path, scalar_reader, fast_figures, monkeypatch):
    root = tmp_path / "source"
    loaded = dict(report._SOURCE_BYTES)
    read_bytes = Path.read_bytes

    def disk_edit(path):
        return b"later disk edit" if path.resolve() == Path(report.__file__).resolve() else read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", disk_edit)
    generation = report.build_report(root, config())
    for name, content in loaded.items():
        assert (root / "report" / generation["generation"] / name).read_bytes() == content


def test_selection_and_configuration_cannot_override_sealed_inputs(tmp_path, scalar_reader):
    root = tmp_path / "selection"
    identity(root, config())
    with pytest.raises(ValueError, match="differs from sealed"):
        report.build_report(root, config(seed=9))
    frozen = selection()
    scalar_reader["decision"] = frozen
    changed = deepcopy(frozen)
    changed["recipes"]["B"]["ridge"] = 10.
    write_json(root / "selection.json", changed)
    with pytest.raises(ValueError, match="sealed calibration decision"):
        report.build_report(root)
    assert not (root / "report/latest.json").exists()


def test_progress_and_failed_calibration_report_complete_outcome_without_confirmation():
    rows = records("development", replicas=range(4))
    rows[-1]["metadata"]["imported"] = False
    rows[-1]["metadata"]["origin_root"] = None
    source = snapshot(rows, selection=None, status=dict(status="calibration_infeasible", phase="7C",
        calibration_status="calibration_infeasible", proposal_round=3, active_seconds=21*3600,
        baseline_seconds=19*3600, incremental_allowance_seconds=72000))
    result = report._summarize(source)
    assert result["status"] == "calibration_infeasible"
    assert not result["calibration"]["confirmation_authorized"]
    text = report._progress(source, result)
    assert "imported 19, new 1" in text and "proposal round 3" in text
    assert "2 / 20 h" in text and "fresh 0/8 paired blocks" in text
    assert "Fresh confirmation is not authorized" in report._conclusion(source, result)


def test_interruption_lease_tampering_and_allocation_bounds(tmp_path, scalar_reader, fast_figures, monkeypatch):
    root = tmp_path / "atomic"
    first = report.build_report(root, config())
    pointer = (root / "report/latest.json").read_bytes()
    atomic = original._atomic

    def interrupt(path, data):
        if path.name == "analysis.json":
            raise InterruptedError("stop requested")
        return atomic(path, data)

    monkeypatch.setattr(original, "_atomic", interrupt)
    with pytest.raises(InterruptedError):
        report.build_report(root, config(), force=True)
    assert (root / "report/latest.json").read_bytes() == pointer
    assert not list((root / "report").glob(".tmp-generation-*"))
    monkeypatch.setattr(original, "_atomic", atomic)
    with (root / "report/.publish.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            report.build_report(root, config())
    png = root / "report" / first["generation"] / "figures/development.png"
    png.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="generation changed"):
        report.build_report(root, config())
    monkeypatch.setattr(report, "MAX_JSON_BYTES", 1)
    with pytest.raises(ValueError, match="bounded report JSON"):
        report.build_report(tmp_path / "bounded", config())
