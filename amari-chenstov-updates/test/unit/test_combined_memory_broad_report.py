"""Bounded Phase 5 reporting tests using sealed scalar fixtures, never models."""

from copy import deepcopy
import json
from pathlib import Path

import nbformat
import numpy as np
import pytest

from mnist_experiment.combined_memory import analysis, report as original
from mnist_experiment.combined_memory.broad_drift import report


def config(**changes):
    return dict(protocol=report.PROTOCOL, smoke=True, analysis_novel_count=4,
        samples_per_step=2, hold_steps=1, ramp_steps=2, stream_size=12,
        bootstrap_draws=24, seed=18, main_replicas=2, stationary_replicas=1,
        retention_tolerance=.1, retention_ceiling=.5, **changes)


def record(arm="A", replica=0, cohort="fresh", cfg=None):
    cfg = cfg or config()
    m, hold, ramp = (cfg[k] for k in ("samples_per_step", "hold_steps", "ramp_steps"))
    total = 2 * (hold + ramp)
    endpoint = dict(A=8., B=6., C=5., D=4., E=3.)[arm]
    rows = []
    for step in range(total + 1):
        n = 0 if cohort == "stationary" else m * max(0, min(step - hold, ramp))
        risk = 20. if step == 0 else 16. - (16. - endpoint) * n / cfg["analysis_novel_count"]
        old = 4. if step == 0 else 4.0401
        if step > hold + ramp:
            risk, old = 999., 900.  # Return risk must never overwrite the endpoint.
        rows.append(dict(accepted_step=step, incoming_grids=m * step,
            novel_count=n, outbound_count=n, kind="initial" if step == 0 else "regular",
            novel_mse=risk, old_mse=old, current_mse=1. + step / total,
            old_accuracy=.75, novel_accuracy=.5, current_accuracy=.7,
            frozen_current_mse=3., learning_seconds=step * (2. if arm == "E" else 1.)))
    summary = analysis.summarize_path(rows, cfg)
    summary.update(H_endpoint=summary["H100"], H1216=summary["H100"] if cfg["analysis_novel_count"] == 1216 else None,
                   old_mse_endpoint=summary["old_mse_100"], rotated_mse_endpoint=summary["novel_mse_100"])
    return dict(unit_id=f"{cohort}-{replica}-{arm}", status="completed", summary=summary,
        metadata=dict(kind="trajectory", cohort=cohort, replica=replica, protocol=report.PROTOCOL,
            recipe=dict(arm=arm, ridge=1., strength=0. if arm == "A" else .2, process_scale=1e-4),
            pairing_asset_hash=f"{cohort}-assets-{replica}"), metrics=rows,
        diagnostics=[dict(batch=2, incoming_grids=2 * m, novel_count=m, outbound_count=m,
            applied_pi=.025, raw_pi=.02, motion_energy=.2, uncertainty_energy=.1, noise_scale=.08,
            kalman_level_gain_mean=.3, kalman_level_gain_p10=.1, kalman_level_gain_p90=.6,
            kalman_velocity_gain_mean=.03, kalman_velocity_gain_p10=.01, kalman_velocity_gain_p90=.06,
            negative_mass=.01, projection_norm=.02, correction_norm=.1)])


def records(cfg=None):
    return [record(a, r, cfg=cfg) for r in range(2) for a in analysis.ARMS]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = original._bytes(value)
    path.write_bytes(data)
    return dict(sha256=original._hash(data), bytes=len(data))


def store(root, rows, cfg=None):
    cfg = cfg or config()
    identity = dict(protocol=report.PROTOCOL, metric_schema_version=1,
                    artifact_schema_version=1, config=cfg,
                    acquisition_clock="unique_outbound_grids; stationary has no acquisition")
    digest = original._hash(original._bytes(identity))
    write_json(root / "identity.json", dict(schema=1, identity=identity, identity_hash=digest))
    for row in rows:
        uid, metadata = row["unit_id"], row["metadata"]
        path = root / "units" / uid
        manifest = dict(schema=1, unit_id=uid, identity_hash=digest, metadata=metadata,
                        metadata_hash=original._hash(original._bytes(metadata)))
        seals = {"manifest.json": write_json(path / "manifest.json", manifest)}
        if row["status"] == "completed":
            for name in ("summary", "metrics", "diagnostics"):
                seals[f"{name}.json"] = write_json(path / f"{name}.json", row[name])
            # A checkpoint may exist, but reporting must not read or hash it.
            (path / "checkpoint.pt").write_bytes(b"not a torch payload")
            seals["checkpoint.pt"] = dict(sha256="intentionally not inspected", bytes=19)
            marker = dict(schema=1, unit_id=uid, identity_hash=digest, files=seals)
            marker["seal_hash"] = original._hash(original._bytes(marker))
            write_json(path / "COMPLETED", marker)


@pytest.fixture
def history(tmp_path, monkeypatch):
    root = tmp_path / "production_v1"
    book = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell("# Original Phase 0--4\n\nHistorical qualifications remain intact."),
        nbformat.v4.new_code_cell("1 + 1", execution_count=1,
            outputs=[nbformat.v4.new_output("execute_result", data={"text/plain": "2"}, execution_count=1)]),
        nbformat.v4.new_markdown_cell("## Phase 4\n\nOriginal conclusion and null findings.")],
        metadata={"plan18_report": {"protocol": original.PROTOCOL}, "custom": "preserve"})
    generation, digest = "original-generation", "original-source-digest"
    path = root / "report" / generation
    path.mkdir(parents=True)
    data = nbformat.writes(book).encode()
    (path / "results.ipynb").write_bytes(data)
    metadata = dict(schema=1, protocol=original.PROTOCOL, status="completed", generation=generation)
    write_json(path / "generation.json", metadata)
    write_json(path / "COMPLETED", dict(schema=1, source_digest=digest, files={
        "results.ipynb": original._hash(data),
        "generation.json": original._hash((path / "generation.json").read_bytes())}))
    write_json(root / "report/latest.json", dict(schema=1, generation=generation, source_digest=digest))
    monkeypatch.setattr(report, "HISTORY_ROOT", root)
    return root, book


@pytest.fixture
def fast_figures(monkeypatch):
    def render(snapshot, result):
        figure, axes = original._figure(1, 1)
        axes[0].plot([0, 1], [0, 1])
        return figure
    for name in ("_process", "_development", "_prediction", "_diagnostics"):
        monkeypatch.setattr(report, name, render)


def exported(root, generation, name="analysis.json"):
    return json.loads((root / "report" / generation / name).read_bytes())


def test_real_renderers_publish_history_and_four_figures_with_correct_labels(tmp_path, history):
    root, public = tmp_path / "broad", tmp_path / "public.ipynb"
    public.write_text("public must not be consulted")
    cfg = config()
    rows = records() + [record(a, cohort="stationary") for a in analysis.ARMS]
    store(root, rows, cfg)
    write_json(root / "readiness.json", dict(initializations=[], numerical_policy=dict(fd_dtype="float64"),
        detectability=dict(replicas=[dict(replica=r, states=[dict(outbound_batch=1, **{
            law: dict(motion_energy=.2, split_motion_energy=-.1, batch_noise=.02,
                      motion_to_batch_noise=10., split_motion_to_batch_noise=-5., motion_ci=[-.2, .3])
            for law in ("rare", "broad")})]) for r in (0, 1)])))
    generation = report.build_report(root, cfg, notebook_path=public)
    book = nbformat.read(root / "smoke_results.ipynb", as_version=4)
    nbformat.validate(book)
    assert book.cells[1:1 + len(history[1].cells)] == history[1].cells
    assert book.metadata.custom == "preserve"
    assert public.read_text() == "public must not be consulted"
    assert generation["figure_count"] == 4
    phase5 = book.cells[1 + len(history[1].cells):]
    assert len(phase5) == 10
    assert len([c for c in phase5 if c.get("attachments")]) == 4
    assert all(c.cell_type == "markdown" for c in phase5)
    text = "\n".join(c.source for c in phase5)
    assert "H4" in text and "H100" not in text and "Equal-4" in text
    assert "may have diluted" in text and "Stationary paths have no acquisition clock or S" in text
    result = exported(root, generation["generation"])
    assert result["protocol"] == report.PROTOCOL
    assert result["units"]["novel"] == "unique_outbound_grids"
    assert list(result["comparisons"])[0] == "B-A"  # Sorted JSON puts B-A first too.
    assert result["cohorts"]["stationary"]["efficiency"] == {}
    assert result["methods"]["E"]["H_endpoint"] == result["methods"]["E"]["H100"]


def test_dynamic_1216_analysis_matches_original_and_adds_planned_B_A():
    cfg = dict(config(), smoke=False, analysis_novel_count=1216, samples_per_step=4,
               hold_steps=160, ramp_steps=304, stream_size=3712)
    rows = records(cfg)
    snapshot = dict(records=rows, cfg=cfg, selection=None)
    result = report._summarize(snapshot)
    base = analysis.summarize(rows, cfg)
    original._add_incumbent_contrast(snapshot, base)
    assert list(result["comparisons"])[0] == "B-A"
    assert result["comparisons"]["B-A"]["novel_auc"] == base["comparisons"]["B-A"]["novel_auc"]
    assert result["methods"]["A"]["curves"] == base["methods"]["A"]["curves"]
    assert result["methods"]["A"]["curves"]["novel_mse"]["mean"][-1] == 8.
    assert result["methods"]["E"]["H1216"] == result["methods"]["E"]["H100"]
    text = report._effects(result) + report._efficiency(result)
    assert "H1216" in text and "Equal-1216" in text
    assert "H100" not in text and "Equal-100" not in text
    assert "Planned Phase 5" in result["report_additions"]["B-A"]


def test_reference_clock_exposes_identical_policy_apparent_endpoint_saving():
    cfg = config()
    rows = records(cfg)
    for item in rows:
        if item["metadata"]["recipe"]["arm"] not in ("A","B"):
            continue
        for metric in item["metrics"]:
            if metric["accepted_step"] <= cfg["hold_steps"]+cfg["ramp_steps"]:
                n = metric["novel_count"]
                metric["novel_mse"] = 20. if metric["kind"] == "initial" else (
                    16. if n == 0 else 5. if n == 2 else 8.)
        item["summary"] = analysis.summarize_path(item["metrics"],cfg)
    result = report._summarize(dict(records=rows,cfg=cfg,selection=None))
    reference = result["reference_clock_diagnostics"]["A"]
    candidate = result["efficiency"]["B-A"]["point"]
    assert candidate["saved_fraction"] > 0  # Retain the protocol's allocated-budget statistic.
    assert reference["n"] == pytest.approx(candidate["n"])
    assert reference["n"] < reference["allocated_n"]
    assert "does not establish" in report._efficiency(result)


def test_completed_report_distinguishes_unestimable_statistics_from_pending_work():
    cfg = config()
    rows = records(cfg) + [record(a, cohort="stationary", cfg=cfg) for a in analysis.ARMS]
    for row in rows:
        if row["metadata"]["recipe"]["arm"] == "B":
            for metric in row["metrics"]:
                if metric["kind"] != "initial":
                    metric["novel_mse"] += 50.
            row["summary"] = analysis.summarize_path(row["metrics"], cfg)
    selection = {r["metadata"]["recipe"]["arm"]: r["metadata"]["recipe"] for r in rows}
    snapshot = dict(records=rows, cfg=cfg, selection=selection,
        status=dict(status="completed", phase="5E: inference and closeout",
                    active_seconds=60., forecast_remaining_seconds=0., total_paths=15))
    result = report._summarize(snapshot)
    assert result["status"] == "completed"
    assert result["efficiency"]["B-A"]["point"]["status"] == "target_not_reached_at_endpoint"
    progress = report._progress(snapshot, result)
    assert "none (execution complete)" in progress
    assert "no further computation is queued" in progress
    assert "pending" not in progress.lower()
    efficiency = report._efficiency(result)
    assert "| B-A common A | not estimable | not estimable |" in efficiency
    assert "pending" not in efficiency.lower()
    assert "matched-budget retention bound" in efficiency


def test_development_health_uses_latest_complete_recipe_without_fresh_inference():
    first = record("D",0,"development")
    second = record("D",0,"development")
    second["unit_id"] += "-new"
    first["completed_at"],second["completed_at"] = "2026-11-07T01:00:00Z","2026-11-07T02:00:00Z"
    second["metadata"]["recipe"]["ridge"] = .1
    second["diagnostics"][0]["applied_pi"] = .004
    snapshot = dict(records=[second,first],cfg=config(),selection=None)
    result = report._summarize(snapshot)
    assert result["methods"] == {} and result["comparisons"] == {}
    chosen = report._health_records(snapshot,result)
    assert len(chosen) == 1 and chosen[0]["unit_id"] == second["unit_id"]
    figure = report._diagnostics(snapshot,result)
    try:
        series = {line.get_label():line.get_ydata() for line in figure.axes[0].lines}
        assert series["D applied pi"] == pytest.approx([.004])
    finally:
        figure.clear()


@pytest.mark.parametrize("unmatched", [False, True])
def test_conclusion_distinguishes_failed_calibration_from_method_failure(unmatched):
    rows = records() + [record(a, cohort="stationary") for a in analysis.ARMS]
    recipes = {r["metadata"]["recipe"]["arm"]: r["metadata"]["recipe"] for r in rows}
    selection = dict(recipes=recipes,
        matching={a: 4.2 if unmatched and a != "B" else 0. for a in analysis.ARMS})
    snapshot = dict(records=rows, cfg=config(), selection=selection)
    result = report._summarize(snapshot)
    before = deepcopy(result)
    text = report._conclusion(snapshot, result)
    assert "Phase 5 Conclusion" in text
    assert ("Calibration failed" in text) == unmatched
    if unmatched:
        assert "factorial interaction" in text
        assert "Other contrasts require their own retention checks" in text
        assert "proposed follow-up, not an executed repair" in text
        assert "earlier report generations are preserved" in text
    assert result == before


def test_split_motion_and_live_noise_remain_linear_and_gain_units_are_distinct(tmp_path):
    snapshot = dict(records=records(), cfg=config(), selection=None,
        readiness=dict(detectability=dict(replicas=[dict(replica=0, states=[dict(outbound_batch=1,
            **{law: dict(motion_energy=1., split_motion_energy=-.5, batch_noise=.1,
                motion_to_batch_noise=10., split_motion_to_batch_noise=-5., motion_ci=[-1., 2.])
               for law in ("rare", "broad")})]) for _ in range(1)])))
    result = report._summarize(snapshot)
    process = report._process(snapshot, result)
    diagnostic = report._diagnostics(snapshot, result)
    development = report._development(snapshot, result)
    try:
        axis = process.axes[2]
        assert axis.get_yscale() == "linear"
        assert any(np.array_equal(line.get_ydata(), [-.5]) for line in axis.lines)
        assert len(axis.get_legend_handles_labels()[0]) <= 5
        assert diagnostic.axes[0].get_ylabel() == r"$\pi$"
        assert "level gain" in diagnostic.axes[1].get_title()
        noise = {line.get_label(): line.get_ydata() for line in diagnostic.axes[3].lines}
        assert "A one-grid V" not in noise
        assert noise["D one-grid V"] == pytest.approx([.08])
        assert noise["D batch V/m"] == pytest.approx([.04])
        assert noise["D motion + uncertainty"] == pytest.approx([.3])
        assert "H4" in development.axes[1].get_xlabel()
    finally:
        for figure in (process, diagnostic, development):
            figure.clear()


def test_immutable_rebuilds_preserve_history_and_completed_science(tmp_path, history, fast_figures):
    root, public = tmp_path / "broad", tmp_path / "public.ipynb"
    cfg = dict(config(), smoke=False, analysis_novel_count=1216, samples_per_step=4,
               hold_steps=160, ramp_steps=304, stream_size=3712)
    store(root, [], cfg)
    before = {p: p.read_bytes() for p in history[0].rglob("*") if p.is_file()}
    identity = (root / "identity.json").read_bytes()
    first = report.build_report(root, cfg, notebook_path=public)
    frozen = (root / "report" / first["generation"] / "results.ipynb").read_bytes()
    assert not report.build_report(root, cfg, notebook_path=public)["changed"]
    public.write_text("mutable public corruption must not become history")
    assert not report.build_report(root, cfg, notebook_path=public)["changed"]
    assert public.read_bytes() == frozen
    forced = report.build_report(root, cfg, notebook_path=public, force=True)
    assert forced["generation"] != first["generation"]
    assert report.build_report(root, cfg, notebook_path=public)["generation"] == forced["generation"]
    write_json(root / "status.json", dict(phase="5C", active_seconds=3600,
        forecast_remaining_seconds=7200, current=dict(unit="init", batch=2, total=6, epoch=3)))
    updated = report.build_report(root, cfg, notebook_path=public)
    book = nbformat.read(public, as_version=4)
    assert "5C" in book.cells[0].source and "2/6" in book.cells[0].source
    assert "initialization epoch 3" in book.cells[0].source
    assert "34 paths" in book.cells[0].source and "provisional" in book.cells[0].source
    assert book.cells[1:1 + len(history[1].cells)] == history[1].cells
    assert sum(c.source.startswith("## Phase 5:") for c in book.cells) == 1
    assert (root / "report" / first["generation"] / "results.ipynb").read_bytes() == frozen
    assert updated["generation"] != first["generation"]
    assert (root / "identity.json").read_bytes() == identity
    assert all(p.read_bytes() == data for p, data in before.items())


def test_report_archives_imported_sources_even_after_on_disk_edits(tmp_path, history, fast_figures, monkeypatch):
    root = tmp_path/"source-provenance"
    loaded = dict(report._SOURCE_BYTES)
    implementation = Path(report.__file__).resolve()
    read_bytes = Path.read_bytes

    def changed_after_import(path):
        return b"different on-disk presentation" if path.resolve() == implementation else read_bytes(path)

    monkeypatch.setattr(Path,"read_bytes",changed_after_import)
    generation = report.build_report(root,config())
    directory = root/"report"/generation["generation"]
    inputs = json.loads((directory/"inputs.json").read_text())
    assert inputs["implementation"]["broad_report"] == original._hash(loaded["report_source.py"])
    marker = json.loads((directory/"COMPLETED").read_text())
    for name,content in loaded.items():
        assert (directory/name).read_bytes() == content
        assert marker["files"][name] == original._hash(content)


def test_partial_blocks_are_progress_not_inference(tmp_path, history):
    rows = records()
    rows[-1] = dict(rows[-1], status="incomplete")
    root = tmp_path / "partial"
    store(root, rows)
    snapshot = report._snapshot(root, config(), lambda: None)
    result = report._summarize(snapshot)
    assert result["cohorts"]["fresh"]["replicas"] == [0]
    assert result["cohorts"]["fresh"]["partial_blocks"] == [dict(replica=1, complete_arms=list("ABCD"))]
    assert result["comparisons"]["B-A"]["novel_auc"]["n_pairs"] == 1
    assert result["comparisons"]["B-A"]["novel_auc"]["ci"] is None


@pytest.mark.parametrize("corruption", ["protocol", "seal", "missing", "clock", "summary"])
def test_incompatible_scalar_inputs_fail_without_repair(tmp_path, corruption):
    root, rows = tmp_path / "bad", records()
    if corruption == "clock":
        rows[0]["metrics"][-1]["outbound_count"] += 1
    if corruption == "summary":
        rows[0]["summary"]["H100"] += 1
    store(root, rows)
    if corruption == "protocol":
        path = root / "identity.json"
        value = json.loads(path.read_bytes())
        value["identity"]["protocol"] = original.PROTOCOL
        value["identity_hash"] = original._hash(original._bytes(value["identity"]))
        write_json(path, value)
    if corruption in ("seal", "missing"):
        path = root / "units/fresh-0-A/metrics.json"
        if corruption == "seal":
            path.write_text("[]")
        else:
            path.unlink()
    with pytest.raises(ValueError, match="protocol|seal|regular file|clock|summary disagrees"):
        report._summarize(report._snapshot(root, config(), lambda: None))
    assert not (root / "report/latest.json").exists()


def test_smoke_without_history_is_local_and_pending(tmp_path, monkeypatch, fast_figures):
    monkeypatch.setattr(report, "HISTORY_ROOT", tmp_path / "absent-history")
    public = tmp_path / "public.ipynb"
    public.write_text("historical public bytes")
    root = tmp_path / "smoke"
    first = report.build_report(root, config(), notebook_path=public)
    assert first["status"] == "pending_or_partial"
    assert public.read_text() == "historical public bytes"
    assert (root / "smoke_results.ipynb").exists()
    assert not report.build_report(root, config(), notebook_path=public)["changed"]
    assert not (root / "identity.json").exists()
    with pytest.raises(ValueError, match="history is required"):
        report.build_report(tmp_path / "production", dict(config(), smoke=False, analysis_novel_count=1216), notebook_path=public)


def test_input_and_generation_bounds_and_tamper_checks(tmp_path, history, fast_figures, monkeypatch):
    root = tmp_path / "bounded"
    write_json(root / "status.json", dict(padding="x" * 1000))
    monkeypatch.setattr(report, "MAX_JSON_BYTES", 512)
    with pytest.raises(ValueError, match="JSON input allocation"):
        report.build_report(root, config())
    monkeypatch.setattr(report, "MAX_JSON_BYTES", original.MAX_JSON_BYTES)
    first = report.build_report(root, config())
    png = root / "report" / first["generation"] / "figures/process.png"
    png.write_bytes(b"changed")
    with pytest.raises(ValueError, match="generation changed"):
        report.build_report(root, config())
    assert png.read_bytes() == b"changed"
    assert report.MAX_JSON_BYTES >= 78 * 2 * 1024**2


def test_interruption_cannot_publish_partial_generation(tmp_path, history, fast_figures, monkeypatch):
    root = tmp_path / "interrupt"
    first = report.build_report(root, config())
    before = (root / "report/latest.json").read_bytes()
    book = (root / "smoke_results.ipynb").read_bytes()
    atomic = original._atomic

    def interrupted(path, data):
        if path.name == "analysis.json":
            raise InterruptedError("stop")
        atomic(path, data)

    monkeypatch.setattr(original, "_atomic", interrupted)
    with pytest.raises(InterruptedError):
        report.build_report(root, config(), force=True)
    assert (root / "report/latest.json").read_bytes() == before
    assert (root / "smoke_results.ipynb").read_bytes() == book
    assert (root / "report" / first["generation"] / "COMPLETED").exists()
    assert not list((root / "report").glob(".tmp-generation-*"))
