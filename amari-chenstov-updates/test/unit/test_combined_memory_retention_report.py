"""Scalar-only Phase 6 inference, chronology, and immutable publication tests."""

from copy import deepcopy
import fcntl
import json
from pathlib import Path

import nbformat
import numpy as np
import pytest

from mnist_experiment.combined_memory import analysis, report as original
from mnist_experiment.combined_memory.broad_drift import report as broad
from mnist_experiment.combined_memory.retention_match import report, runtime


def config(**changes):
    return dict(dict(protocol=report.PROTOCOL, smoke=True, analysis_novel_count=4,
        samples_per_step=2, hold_steps=1, ramp_steps=2, stream_size=12,
        bootstrap_draws=32, seed=18110806, main_replicas=4, stationary_replicas=2,
        retention_tolerance=.1), **changes)


def recipe(arm):
    return dict(arm=arm, ridge=dict(A=10., B=10., C=10., D=1., E=.1)[arm],
                strength=0. if arm == "A" else .5691316767656495,
                process_scale=1e-5 if arm in "DE" else 1e-4)


def selection():
    return dict(frozen=True, recipes={arm: recipe(arm) for arm in analysis.ARMS},
        main_replicas=4, stationary_replicas=2, bootstrap_seed=196,
        calibration={}, matching=dict(passed=True, ranges=dict(Havg=.04, H100=.04), means={}),
        members=[0, 1, 2, 3])


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
            risk, old = 999., 900.  # Repeated outbound counts must not replace acquisition endpoints.
        rows.append(dict(accepted_step=step, incoming_grids=m * step, outbound_count=n,
            novel_count=n, kind="initial" if step == 0 else "regular", novel_mse=risk, old_mse=old,
            current_mse=1. + step / total, old_accuracy=.75, novel_accuracy=.5, current_accuracy=.7,
            frozen_current_mse=3., learning_seconds=step * (2. if arm == "E" else 1.)))
    return dict(unit_id=f"{cohort}-{replica}-{arm}", status="completed", metrics=rows,
        summary=analysis.summarize_path(rows, cfg), diagnostics=[],
        metadata=dict(kind="trajectory", cohort=cohort, replica=replica, protocol=report.PROTOCOL,
            recipe=recipe(arm), pairing_asset_hash=f"{cohort}-assets-{replica}",
            imported=cohort == "development" and replica in (0, 1)))


def records(cohort="fresh", cfg=None, replicas=range(4)):
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
        value = state.get("decision", json.loads((root / "selection.json").read_bytes()))
        return value, original._hash(original._bytes(value))

    monkeypatch.setattr(runtime, "read_exported_records", read)
    monkeypatch.setattr(runtime, "checked_scalar", decision)
    return state


@pytest.fixture
def history(tmp_path, monkeypatch):
    root = tmp_path / "broad_drift_v1"
    book = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Original Phase 0--4\n\nSealed earlier qualifications."),
        nbformat.v4.new_code_cell("1 + 1", execution_count=1,
            outputs=[nbformat.v4.new_output("execute_result", data={"text/plain": "2"}, execution_count=1)]),
        nbformat.v4.new_markdown_cell("## Phase 5: Broader Continuous Drift\n\nUnchanged numerical results."),
        nbformat.v4.new_markdown_cell("### Phase 5 Conclusion\n\n**Calibration failed:** not comparable retention. "
            "This is not evidence of a general STM failure.")],
        metadata=dict(custom="preserve", plan18_phase5_report=dict(protocol=broad.PROTOCOL)))
    path = root / "report" / "sealed-phase5"
    path.mkdir(parents=True)
    (path / "results.ipynb").write_bytes(nbformat.writes(book).encode())
    write_json(path / "generation.json", dict(schema=1, protocol=broad.PROTOCOL, status="completed"))
    reseal_history(root)
    monkeypatch.setattr(report, "HISTORY_ROOT", root)
    return root, book


def reseal_history(root):
    path = root / "report/sealed-phase5"
    digest = "historical-source-digest"
    write_json(path / "COMPLETED", dict(schema=1, source_digest=digest,
        files={name: original._hash((path / name).read_bytes()) for name in ("results.ipynb", "generation.json")}))
    write_json(root / "report/latest.json", dict(schema=1, generation=path.name, source_digest=digest))


@pytest.fixture
def fast_figures(monkeypatch):
    def render(snapshot, result):
        figure, axes = original._figure(1, 1)
        axes[0].plot([0, 1], [0, 1])
        return figure

    monkeypatch.setattr(report, "_development", render)
    monkeypatch.setattr(report, "_prediction", render)
    monkeypatch.setattr(broad, "_diagnostics", render)


def test_real_renderers_preserve_history_and_publish_compact_complete_scalar_story(tmp_path, history, scalar_reader, monkeypatch):
    import torch

    def forbidden(*args, **kwargs):
        pytest.fail("report attempted model/checkpoint or training access")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(runtime, "load_readonly_checkpoint", forbidden)
    root, public = tmp_path / "retention", tmp_path / "public.ipynb"
    public.write_text("public is not a history source")
    cfg = config()
    identity(root, cfg)
    rows = records("development") + records() + records("stationary", replicas=range(2))
    scalar_reader["records"] = rows
    write_json(root / "selection.json", selection())
    write_json(root / "development.json", dict(members=[0, 1, 2, 3], member_summaries=[]))
    write_json(root / "status.json", dict(protocol=report.PROTOCOL, phase="6D", status="completed",
        calibration_status="passed", active_seconds=18 * 3600, forecast_remaining_seconds=0))
    generation = report.build_report(root, notebook_path=public)
    book = nbformat.read(root / "smoke_results.ipynb", as_version=4)
    nbformat.validate(book)
    assert book.cells[1:1 + len(history[1].cells)] == history[1].cells
    assert book.metadata.custom == "preserve"
    phase6 = book.cells[1 + len(history[1].cells):]
    assert len(phase6) == 9
    assert sum(bool(c.get("attachments")) for c in phase6) == 3
    assert all(c.cell_type == "markdown" for c in phase6)
    text = "\n".join(c.source for c in phase6)
    assert "H4" in text and "H100" not in text
    assert "historically informed" in text
    assert "Reference-clock check" in text and "Stationary paths have no acquisition clock or S" in text
    assert "nominal two-sided 95%" in text and "[-.10, .10]" in text
    assert "Pi, Gain And Motion Diagnostics" in text
    assert "not confidence intervals" in text
    assert "mean trajectories, not a median" in text
    assert "recalculates the control's endpoint target" in text
    assert "development 20/20 (imported development 10" in book.cells[0].source
    assert "18 / 20 h" in book.cells[0].source
    assert public.read_text() == "public is not a history source"
    result = exported(root, generation["generation"])
    assert result["status"] == "completed" and result["protocol"] == report.PROTOCOL
    assert len(result["retention_equivalence"]) == 10
    assert result["cohorts"]["stationary"]["efficiency"] == {}
    assert generation["figure_count"] == 3
    assert (root / "report" / generation["generation"] / "figures/diagnostics.png").is_file()


def test_diagnostics_use_fresh_paired_means_and_keep_noise_units_distinct():
    rows = records("development") + records() + records("stationary", replicas=range(2))
    for row in rows:
        replica = row["metadata"]["replica"]
        offset = replica if row["metadata"]["cohort"] == "fresh" else 1000
        row["diagnostics"] = [dict(incoming_grids=clock,
            applied_pi=.01 * (offset + 1), raw_pi=.02 * (offset + 1),
            kalman_level_gain_mean=.1 * offset, kalman_level_gain_p10=.05 * offset,
            kalman_level_gain_p90=.15 * offset, kalman_velocity_gain_mean=.01 * (offset + 1),
            kalman_velocity_gain_p10=.005 * (offset + 1), kalman_velocity_gain_p90=.015 * (offset + 1),
            noise_scale=10 + offset, motion_energy=2 + offset, uncertainty_energy=1.,
            correction_norm=offset + 3., projection_norm=0.) for clock in (2, 4)]
    source = snapshot(rows)
    result = report._summarize(source)
    figure = broad._diagnostics(source, result)
    try:
        assert len(figure.axes) == 6
        for axis_index, label, expected in (
                (0, "D applied pi", .025), (0, "E raw pi", .05),
                (1, "D level", .15), (2, "E velocity", .025),
                (3, "D one-grid V", 11.5), (3, "D batch V/m", 5.75),
                (3, "E motion + uncertainty", 4.5), (4, "C correction norm", 4.5)):
            line = next(line for line in figure.axes[axis_index].lines if line.get_label() == label)
            np.testing.assert_allclose(line.get_xdata(), [2, 4])
            np.testing.assert_allclose(line.get_ydata(), [expected, expected])
        assert len(figure.axes[1].collections) == 2
        stationary = [line for line in figure.axes[5].lines if len(line.get_xdata()) == 7]
        assert len(stationary) == 10
    finally:
        figure.clear()


def test_all_ten_retention_intervals_reuse_whole_paired_sign_stratified_draws():
    source = snapshot()
    result = report._summarize(source)
    fresh = result["cohorts"]["fresh"]
    draws = analysis.paired_draws(fresh["replicas"], 32, selection()["bootstrap_seed"])
    assert len(result["retention_equivalence"]) == 10
    for name, item in result["retention_equivalence"].items():
        left, right = name.split("-")
        for field in ("Havg", "H100"):
            values = [np.asarray(result["methods"][arm][field]["per_replica"]) for arm in (left, right)]
            expected = analysis._interval(values[0] - values[1], draws)
            assert item[field] == dict(expected, equivalent=True)
        assert item["equivalent"]
    assert list(result["comparisons"])[0] == "B-A"
    inherited = broad._summarize(source)
    assert result["efficiency"] == inherited["efficiency"]
    assert result["reference_clock_diagnostics"] == inherited["reference_clock_diagnostics"]
    assert result["methods"]["A"]["curves"]["novel_mse"]["mean"][-1] == 8.


@pytest.mark.parametrize("replicas", [(0, 1), (0, 1, 2)])
def test_singleton_sign_strata_cannot_produce_interim_inferential_claims(replicas):
    source = snapshot(records(replicas=replicas))
    result = report._summarize(source)
    inherited = broad._summarize(source)
    assert result["interval_status"] == "descriptive_insufficient_sign_replicates"
    assert result["cohorts"]["fresh"]["bootstrap"]["draws"] == 0
    assert result["methods"]["A"]["curves"]["novel_mse"]["mean"] == inherited["methods"]["A"]["curves"]["novel_mse"]["mean"]
    for item in result["comparisons"].values():
        assert all(value["ci"] is None for value in item.values())
    for name, item in result["efficiency"].items():
        assert item["point"] == inherited["efficiency"][name]["point"]
        assert item["ci"] is None
        assert not item["sample_saving_supported"] and not item["retention_supported"]
    assert all(not item["equivalent"] for item in result["retention_equivalence"].values())
    assert "singleton strata" in report._retention(result)


def test_development_legend_identifies_arms_before_all_four_members_complete():
    source = snapshot(rows=records("development", replicas=(0, 1)), selection=None)
    result = report._summarize(source)
    figure = report._development(source, result)
    try:
        assert set(figure.axes[0].get_legend_handles_labels()[1]) == set(analysis.ARMS)
    finally:
        figure.clear()


def test_progress_includes_all_sealed_refinement_recipes_not_just_current_completions():
    source = snapshot(rows=records("development", replicas=(0, 1)), selection=None)
    source["status"].update(expected_development_new=10)
    proposal = {**recipe("E"), "ridge": .31622776601683794}
    source["refinement_decisions"] = [dict(recipes=[proposal])]
    result = report._summarize(source)
    text = report._progress(source, result)
    assert "development 10/24" in text


def test_production_1216_budget_and_table_labels_keep_the_inherited_clock():
    cfg = config(smoke=False, analysis_novel_count=1216, samples_per_step=4,
                 hold_steps=160, ramp_steps=304, stream_size=3712)
    source = snapshot(records(cfg=cfg), cfg)
    result = report._summarize(source)
    inherited = analysis.summarize(source["records"], cfg, selection())
    assert result["methods"]["A"]["curves"] == inherited["methods"]["A"]["curves"]
    assert len(result["methods"]["A"]["curves"]["n"]) == 1217
    assert result["methods"]["A"]["H1216"] == result["methods"]["A"]["H100"]
    text = report._effects(result) + report._retention(result) + broad._efficiency(result)
    assert "H1216" in text and "Equal-1216" in text and "H100" not in text
    assert len(result["retention_equivalence"]) == 10


def test_nonsignificant_retention_is_not_equivalence_and_both_metrics_required():
    rows = records()
    for row in rows:
        if row["metadata"]["recipe"]["arm"] == "B":
            shift = .24 if row["metadata"]["replica"] < 2 else -.24
            for metric in row["metrics"]:
                if metric["accepted_step"] == 3:
                    metric["old_mse"] = (np.sqrt(metric["old_mse"]) + shift) ** 2
            row["summary"] = analysis.summarize_path(row["metrics"], config())
    result = report._summarize(snapshot(rows))
    item = result["retention_equivalence"]["B-A"]
    assert item["H100"]["ci"][0] < 0 < item["H100"]["ci"][1]
    assert not item["H100"]["equivalent"] and not item["equivalent"]
    assert "unresolved" in report._retention(result)


def test_partial_and_incomplete_blocks_are_progress_only():
    rows = records()
    rows[-1]["status"] = "incomplete"
    source = snapshot(rows)
    result = report._summarize(source)
    fresh = result["cohorts"]["fresh"]
    assert fresh["replicas"] == [0, 1, 2]
    assert fresh["partial_blocks"] == [dict(replica=3, complete_arms=list("ABCD"))]
    assert result["retention_equivalence"]["B-A"]["Havg"]["n_pairs"] == 3
    assert result["comparisons"]["B-A"]["novel_auc"]["n_pairs"] == 3
    one = report._summarize(snapshot(records(replicas=[0])))
    assert one["retention_equivalence"]["B-A"]["Havg"]["ci"] is None
    assert not one["retention_equivalence"]["B-A"]["equivalent"]


@pytest.mark.parametrize("status", ["pending", "calibration_infeasible", "calibration_budget_limited"])
def test_empty_or_failed_calibration_publishes_without_fresh_claims(tmp_path, scalar_reader, fast_figures, monkeypatch, status):
    monkeypatch.setattr(report, "HISTORY_ROOT", tmp_path / "no-history")
    root, public = tmp_path / status, tmp_path / "public.ipynb"
    public.write_text("human notebook")
    scalar_reader["records"] = records("development") if status != "pending" else []
    write_json(root / "status.json", dict(protocol=report.PROTOCOL, calibration_status=status, phase="6B"))
    generation = report.build_report(root, config(retention_tolerance=10.), notebook_path=public)
    book = nbformat.read(root / "smoke_results.ipynb", as_version=4)
    result = exported(root, generation["generation"])
    assert not result["calibration"]["passed"] and not result["methods"]
    assert result["retention_equivalence"] == {}
    assert "Fresh confirmation is not authorized" in book.cells[-1].source
    assert status in book.cells[0].source
    assert public.read_text() == "human notebook"


def test_empty_real_runtime_reader_accepts_the_scheduler_smoke_configuration(tmp_path, fast_figures, monkeypatch):
    from mnist_experiment.combined_memory.retention_match.config import Config

    monkeypatch.setattr(report, "HISTORY_ROOT", tmp_path / "absent")
    root = tmp_path / "unstarted"
    generation = report.build_report(root, Config.smoke_config())
    assert generation["status"] == "pending_or_partial"
    assert (root / "smoke_results.ipynb").exists()
    assert not (root / "identity.json").exists()


@pytest.mark.parametrize("status_field", [True, False])
def test_passed_calibration_with_unfunded_confirmation_is_terminal_not_active(tmp_path, scalar_reader, fast_figures, monkeypatch, status_field):
    monkeypatch.setattr(report, "HISTORY_ROOT", tmp_path / "absent")
    root = tmp_path / "budget-closed"
    identity(root, config())
    frozen = selection()
    frozen["calibration_status"] = "passed"
    write_json(root / "selection.json", frozen)
    scalar_reader["records"] = records("development")
    status = dict(protocol=report.PROTOCOL, calibration_status="passed", phase="6B: confirmation reservation")
    if status_field:
        status["status"] = "calibration_budget_limited"
    write_json(root / "status.json", status)
    detail = "The four-fresh/two-stationary allocation requires 3.2 h; only 2.1 h remains."
    write_json(root / "calibration_completed.json", dict(status="calibration_budget_limited", detail=detail))
    generation = report.build_report(root)
    book = nbformat.read(root / "smoke_results.ipynb", as_version=4)
    assert "Calibration: **passed**" in book.cells[0].source
    assert "Confirmation is closed due to budget" in book.cells[0].source
    assert "Development calibration passed" in book.cells[-1].source
    assert detail in book.cells[-1].source
    assert "No further confirmation work is queued" in book.cells[-1].source
    assert "Allocation remains incomplete" not in book.cells[-1].source
    assert "awaits complete" not in book.cells[-1].source
    inputs = exported(root, generation["generation"], "inputs.json")
    assert inputs["files"]["calibration_completed.json"] == original._hash((root / "calibration_completed.json").read_bytes())


@pytest.mark.parametrize("defect", ["status", "matching", "members", "unfrozen"])
def test_fresh_inference_rejects_missing_passing_four_member_gate(defect):
    source = snapshot()
    if defect == "status":
        source["status"]["calibration_status"] = "pending"
    elif defect == "matching":
        source["selection"]["matching"]["passed"] = False
    elif defect == "members":
        source["selection"]["members"] = [0, 1]
    else:
        source["selection"]["frozen"] = False
    with pytest.raises(ValueError, match="passing four-member calibration"):
        report._summarize(source)


def test_development_renderers_separate_historical_new_and_pooled_members():
    source = snapshot(records("development"))
    result = report._summarize(source)
    calibration, prediction = report._development(source, result), report._prediction(source, result)
    try:
        assert len(calibration.axes) == 2
        assert "H4" in calibration.axes[1].get_xlabel()
        assert len(calibration.axes[0].collections) == 25  # Twenty member points, five pooled means.
        old = prediction.axes[2]
        assert sum(line.get_linestyle() == "--" for line in old.lines) == 10
        assert sum(line.get_linestyle() == ":" for line in old.lines) == 10
        assert sum("pooled four" in line.get_label() for line in old.lines) == 5
        text = report._settings(source, result)
        assert "hist/new/pool" in text and "Maximum mean old-RMS separation" in text
    finally:
        calibration.clear()
        prediction.clear()


def test_unfrozen_development_still_shows_starting_tuple_means_without_selection():
    source = snapshot(records("development"), selection=None, status=dict(calibration_status="calibration_infeasible"))
    source["development"] = dict(current_check=dict(passed=False, ranges=dict(Havg=.2, H100=.03)))
    result = report._summarize(source)
    figure = report._development(source, result)
    try:
        assert len(figure.axes[0].collections) == 25
        assert not result["selection"] and not result["methods"]
        text = report._settings(source, result)
        assert "no selection is frozen" in text and "H4=0.03 px" in text
        assert "H100" not in text
        assert "0.0215 / 0.0275 / 0.0245" in text
    finally:
        figure.clear()


def test_progress_counts_all_imported_candidates_and_scheduler_resource_ceiling():
    rows = records("development")
    extra = deepcopy(rows[0])
    extra["unit_id"] += "-extra"
    extra["metadata"]["recipe"]["ridge"] = 100.
    source = snapshot(rows + [extra], status=dict(calibration_status="pending", expected_development_new=10,
        active_seconds=17 * 3600, max_active_seconds=19 * 3600, remaining_seconds=2 * 3600))
    result = report._summarize(source)
    text = report._progress(source, result)
    assert "development 21/21 (imported development 11" in text
    assert "17 / 19 h" in text and "remaining ceiling: 2 h" in text


def test_completed_conclusion_distinguishes_persistent_crossings_from_initial_learning_speed():
    source = snapshot(status=dict(status="completed",calibration_status="passed"))
    result = report._summarize(source)
    result["status"] = "completed"
    result["reference_clock_diagnostics"] = dict(A=dict(multiple_crossings=True,n=1182.,
                                                      downward_crossings=[629.,1182.]))
    result["efficiency"]["E-A"]["point"]["n"] = 619.
    result["efficiency"]["B-A"]["point"]["n"] = 1181.
    text = report._conclusion(source,result)
    assert "earlier *sustained* quality" in text
    assert "not twice-as-fast initial learning" in text
    assert "does not establish an additional timing advantage" in text
    assert "wide interval" in text


@pytest.mark.parametrize("hours", [20., 21.5])
def test_report_records_the_approved_resource_change_only_after_activation(hours):
    source = snapshot(status=dict(calibration_status="passed",max_active_seconds=hours*3600))
    source.update(source_digest="test-source",inputs=dict(history="test-history"))
    result = report._summarize(source)
    book = nbformat.reads(report._notebook(source,result,nbformat.v4.new_notebook(),[]).decode(),as_version=4)
    intro = next(c.source for c in book.cells if c.source.startswith("## Phase 6:"))
    assert ("**Compute authorization:**" in intro) == (hours > 20)
    if hours > 20:
        assert "30-trajectory confirmation allocation unchanged" in intro


def test_production_append_is_immutable_and_public_corruption_never_becomes_history(tmp_path, history, scalar_reader, fast_figures):
    root, public = tmp_path / "production", tmp_path / "public.ipynb"
    cfg = config(smoke=False, analysis_novel_count=1216, samples_per_step=4, hold_steps=160,
                 ramp_steps=304, stream_size=3712)
    identity(root, cfg)
    before = {p: p.read_bytes() for p in history[0].rglob("*") if p.is_file()}
    first = report.build_report(root, notebook_path=public)
    data = public.read_bytes()
    assert not report.build_report(root, notebook_path=public)["changed"]
    public.write_text("unsealed edits are never read as history")
    assert not report.build_report(root, notebook_path=public)["changed"]
    assert public.read_bytes() == data
    forced = report.build_report(root, force=True, notebook_path=public)
    assert forced["generation"] != first["generation"]
    assert report.build_report(root, notebook_path=public)["generation"] == forced["generation"]
    write_json(root / "status.json", dict(protocol=report.PROTOCOL, phase="6B", calibration_status="pending",
        active_seconds=15 * 3600, forecast_remaining_seconds=4 * 3600,
        current=dict(unit="development-2-A", batch=2, total=928)))
    updated = report.build_report(root, notebook_path=public)
    assert updated["generation"] != first["generation"]
    book = nbformat.read(public, as_version=4)
    assert book.cells[1:1 + len(history[1].cells)] == history[1].cells
    assert "6B" in book.cells[0].source and "2/928" in book.cells[0].source
    assert sum(c.source.startswith("## Phase 6:") for c in book.cells) == 1
    assert (root / "report" / first["generation"] / "results.ipynb").read_bytes() == data
    assert all(p.read_bytes() == value for p, value in before.items())


@pytest.mark.parametrize("defect", ["missing", "protocol", "conclusion", "tampered"])
def test_production_requires_sealed_corrected_phase5_history(tmp_path, history, scalar_reader, fast_figures, defect):
    historical = history[0]
    if defect == "missing":
        historical = tmp_path / "missing"
    elif defect == "protocol":
        write_json(historical / "report/sealed-phase5/generation.json", dict(protocol=original.PROTOCOL, status="completed"))
        reseal_history(historical)
    else:
        path = historical / "report/sealed-phase5/results.ipynb"
        book = nbformat.read(path, as_version=4)
        book.cells[-1].source = "### Phase 5 Conclusion\n\nUnqualified conclusion."
        nbformat.write(book, path)
        if defect == "conclusion":
            reseal_history(historical)
    with pytest.raises(ValueError, match="history|Phase 5|generation changed"):
        report.build_report(tmp_path / "production", config(smoke=False, analysis_novel_count=1216),
                            notebook_path=tmp_path / "public.ipynb", history_root=historical)


def test_imported_runtime_provenance_and_actual_imported_sources_are_sealed(tmp_path, history, scalar_reader, fast_figures, monkeypatch):
    root = tmp_path / "source"
    loaded = dict(report._SOURCE_BYTES)
    scalar_reader["provenance"]["imports"] = [dict(unit="old-development", sha256="sealed-reference")]
    read_bytes = Path.read_bytes

    def disk_edit(path):
        return b"later on-disk edit" if path.resolve() == Path(report.__file__).resolve() else read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", disk_edit)
    generation = report.build_report(root, config())
    path = root / "report" / generation["generation"]
    marker = exported(root, generation["generation"], "COMPLETED")
    inputs = exported(root, generation["generation"], "inputs.json")
    assert inputs["scalar_exports"] == scalar_reader["provenance"]
    for name, data in loaded.items():
        assert (path / name).read_bytes() == data
        assert marker["files"][name] == inputs["implementation"][name] == original._hash(data)
    scalar_reader["provenance"]["imports"][0]["sha256"] = "new-reference"
    assert report.build_report(root, config())["generation"] != generation["generation"]


def test_corrupt_scalar_reader_and_incompatible_identity_fail_without_publication(tmp_path, scalar_reader, monkeypatch):
    root = tmp_path / "bad"
    cfg = config()
    identity(root, cfg)
    wrong = dict(cfg, seed=9)
    with pytest.raises(ValueError, match="differs from sealed"):
        report.build_report(root, wrong)
    write_json(root / "status.json", dict(protocol=broad.PROTOCOL))
    with pytest.raises(ValueError, match="status protocol"):
        report.build_report(root, cfg)
    write_json(root / "status.json", dict(protocol=report.PROTOCOL))

    def corrupt(root, check=lambda: None):
        raise ValueError("historical scalar reference changed")

    monkeypatch.setattr(runtime, "read_exported_records", corrupt)
    with pytest.raises(ValueError, match="historical scalar"):
        report.build_report(root, cfg)
    assert not (root / "report/latest.json").exists()


def test_mutable_selection_cannot_override_the_completed_calibration_decision(tmp_path, scalar_reader):
    root = tmp_path / "selection"
    identity(root, config())
    frozen = selection()
    scalar_reader["decision"] = frozen
    altered = deepcopy(frozen)
    altered["recipes"]["B"]["ridge"] = 1000.
    write_json(root / "selection.json", altered)
    with pytest.raises(ValueError, match="sealed calibration decision"):
        report.build_report(root)
    assert not (root / "report/latest.json").exists()


def test_interruption_releases_lease_and_never_publishes_partial_generation(tmp_path, history, scalar_reader, fast_figures, monkeypatch):
    root = tmp_path / "interrupt"
    first = report.build_report(root, config())
    pointer, book = (root / "report/latest.json").read_bytes(), (root / "smoke_results.ipynb").read_bytes()
    atomic = original._atomic

    def interrupt(path, data):
        if path.name == "analysis.json":
            raise InterruptedError("stop requested")
        return atomic(path, data)

    monkeypatch.setattr(original, "_atomic", interrupt)
    with pytest.raises(InterruptedError):
        report.build_report(root, config(), force=True)
    assert (root / "report/latest.json").read_bytes() == pointer
    assert (root / "smoke_results.ipynb").read_bytes() == book
    assert (root / "report" / first["generation"] / "COMPLETED").exists()
    assert not list((root / "report").glob(".tmp-generation-*"))
    monkeypatch.setattr(original, "_atomic", atomic)
    assert report.build_report(root, config(), force=True)["changed"]


def test_bounds_tampering_and_report_lease(tmp_path, history, scalar_reader, fast_figures, monkeypatch):
    root = tmp_path / "bounded"
    write_json(root / "status.json", dict(padding="x" * 1000))
    monkeypatch.setattr(report, "MAX_JSON_BYTES", 512)
    with pytest.raises(ValueError, match="JSON input allocation"):
        report.build_report(root, config())
    monkeypatch.setattr(report, "MAX_JSON_BYTES", original.MAX_JSON_BYTES)
    monkeypatch.setattr(report, "MAX_GENERATION_BYTES", 64)
    with pytest.raises(ValueError, match="generation allocation"):
        report.build_report(root, config())
    assert not (root / "report/latest.json").exists()
    monkeypatch.setattr(report, "MAX_GENERATION_BYTES", original.MAX_GENERATION_BYTES)
    first = report.build_report(root, config())
    with (root / "report/.publish.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            report.build_report(root, config())
        fcntl.flock(lock, fcntl.LOCK_UN)
    png = root / "report" / first["generation"] / "figures/development.png"
    png.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="generation changed"):
        report.build_report(root, config())
    assert png.read_bytes() == b"tampered"
