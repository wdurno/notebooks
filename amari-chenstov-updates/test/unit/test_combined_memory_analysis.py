"""CPU-only Plan 18 scalar science and publication; no MNIST or model work."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import nbformat
import numpy as np
import pytest

from mnist_experiment.combined_memory import analysis
from mnist_experiment.combined_memory import report


def config(**changes):
    return SimpleNamespace(analysis_novel_count=4, samples_per_step=2, hold_steps=1, ramp_steps=2,
        stream_size=12, bootstrap_draws=24, seed=18, main_replicas=4, stationary_replicas=2,
        retention_tolerance=.1, retention_ceiling=.5, **changes)


def metric(step, n, risk, *, old=4., current=None, kind="regular", seconds=None):
    return dict(accepted_step=step, incoming_grids=2 * step, novel_count=n, kind=kind,
        novel_mse=float(risk), old_mse=float(old), current_mse=current,
        novel_accuracy=.5, old_accuracy=.75, current_accuracy=None if current is None else .7,
        old_nll=10., novel_nll=20., current_nll=None if current is None else 15.,
        learning_seconds=float(step if seconds is None else seconds),
        frozen_current_mse=None if current is None else current + 1)


def path(arm="A", replica=0, *, stationary=False, end=None):
    # Quiet training changes old risk before the first novel batch. Count 2 is
    # revisited later, but its first post-novel checkpoint must remain fixed.
    endpoint = {"A": 8., "B": 6., "C": 5., "D": 4., "E": 3.}[arm] if end is None else end
    halfway = (16 + endpoint) / 2
    old = 4. if arm == "A" else 4.0401
    n = (0, 0, 0, 0, 0, 0, 0) if stationary else (0, 0, 0, 2, 2, 4, 5)
    rows = [metric(0, n[0], 20., old=4., current=1., kind="initial"),
            metric(1, n[1], 16., old=9., current=2.),
            metric(1, n[2], 16., old=9., kind="zero_novel"),
            metric(2, n[3], halfway, old=old, current=3.),
            metric(3, n[4], halfway + 1, old=16., current=4.),
            metric(4, n[5], endpoint, old=old, current=5.),
            metric(6, n[6], endpoint + .1, old=old, current=7.)]
    if stationary:
        rows = [r for r in rows if r["kind"] != "zero_novel"]
    for row in rows:
        row["learning_seconds"] *= 2 if arm == "E" else 1
    return rows


def record(arm="A", replica=0, cohort="fresh", **kwargs):
    return dict(unit_id=f"{cohort}-{replica}-{arm}", status="completed",
        metadata=dict(kind="trajectory", cohort=cohort, replica=replica,
            recipe=dict(arm=arm, ridge=1., strength=0. if arm == "A" else .2, process_scale=1e-4),
            pairing_asset_hash=f"{cohort}-assets-{replica}"),
        metrics=path(arm, replica, stationary=cohort == "stationary", **kwargs),
        diagnostics=[dict(batch=1, incoming_grids=2, applied_pi=.025, raw_pi=.02,
            motion_energy=.2, uncertainty_energy=.1, noise_scale=.03, rho=.5,
            boundary_high=False, boundary_low=False, negative_mass=.01, projection_norm=.02,
            ltm_move=.1, stm_move=.02)])


def cohort(replicas=range(4), role="fresh"):
    return [record(a, r, role) for r in replicas for a in analysis.ARMS]


def test_health_plot_names_pi_explicitly():
    rows, cfg = cohort(), config()
    figure = report._health(dict(records=rows, cfg=cfg), analysis.summarize(rows, cfg))
    try:
        assert figure.axes[0].get_ylabel() == r"$\pi$"
        assert figure.axes[0].get_title() == r"Raw and applied $\pi$"
    finally:
        figure.clear()


def test_report_input_budget_covers_full_production_cohort():
    # 34 development + four refinements + 40 fresh/stationary paths, with headroom.
    assert report.MAX_JSON_BYTES >= 78 * 2 * 1024**2 + 1024**2


def test_report_input_budget_remains_enforced(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text(json.dumps({"padding": "x" * 600}))
    monkeypatch.setattr(report, "MAX_JSON_BYTES", 512)
    with pytest.raises(ValueError, match="JSON input allocation exceeded"):
        report._snapshot(tmp_path, config(), lambda: None)
    monkeypatch.setattr(report, "MAX_JSON_BYTES", 1024)
    snapshot = report._snapshot(tmp_path, config(), lambda: None)
    assert snapshot["status"]["padding"] == "x" * 600
    assert "status.json" in snapshot["inputs"]["files"]


def test_novel_axis_keeps_first_batch_and_pre_novel_zero_not_initial():
    cfg = config()
    curve = analysis.novel_curve(path(), cfg)
    assert curve["supported"]
    assert curve["observed_counts"] == [0, 2, 4]
    assert curve["observed_steps"] == [1, 2, 4]
    assert curve["novel_mse"] == pytest.approx([16, 14, 12, 10, 8])
    assert curve["learning_seconds"] == pytest.approx([1, 1.5, 2, 3, 4])
    assert curve["incoming_grids"] == pytest.approx([2, 3, 4, 6, 8])
    summary = analysis.summarize_path(path(), cfg)
    assert summary["novel_auc"] == pytest.approx(11.)
    assert summary["old_auc"] == pytest.approx((6.5 + 4 + 4 + 4) / 4)
    assert summary["Havg"] == pytest.approx(np.sqrt(summary["old_auc"]) - 2)
    assert summary["H100"] == 0
    assert summary["retention_avg"] == summary["Havg"]
    assert summary["retention_100"] == summary["H100"]
    assert summary["learning_seconds_100"] == 4
    assert summary["novel_nll_auc"] == 20.


def test_current_trapezoid_uses_allgrid_clock_and_ignores_extra_novel_checkpoints():
    rows = path()
    # Add an off-cadence fixed-panel-only measurement. Its huge losses cannot
    # change the regular current-law integral.
    rows.insert(5, metric(3, 2, 999., kind="novel"))
    summary = analysis.summarize_path(rows, config())
    assert summary["current_whole_mse"] == pytest.approx(4.)
    assert summary["current_whole_rms"] == pytest.approx(2.)
    assert summary["current_outbound_mse"] == pytest.approx(3.)
    assert summary["current_return_mse"] == pytest.approx(5.)
    assert summary["current_initial_quiet_mse"] == pytest.approx(1.5)
    assert summary["current_final_quiet_mse"] == pytest.approx(6.5)
    assert summary["current_quiet_mse"] == pytest.approx(4.)
    assert summary["current_ramp_mse"] == pytest.approx(4.)


def test_stationary_and_under_supported_paths_never_extrapolate():
    stationary = analysis.summarize_path(path(stationary=True), config())
    assert not stationary["novel_supported"]
    assert stationary["novel_auc"] is None and stationary["H100"] is None
    assert stationary["current_whole_rms"] == pytest.approx(2.)
    short = analysis.summarize_path(path()[:5], config())
    assert short["novel_mse_100"] is None and short["current_whole_mse"] is None
    assert analysis.novel_curve(path()[:5], config())["novel_mse"] == [16, 14, 12, None, None]


def test_missing_pre_first_novel_state_is_not_replaced_by_frozen_initialization():
    rows = path()
    rows = [rows[0], *rows[3:]]
    curve = analysis.novel_curve(rows, config())
    assert not curve["supported"] and curve["novel_mse"][0] is None
    assert analysis.summarize_path(rows, config())["novel_auc"] is None


@pytest.mark.parametrize("field,value", [("novel_mse", float("nan")), ("old_mse", -1.),
    ("old_accuracy", 1.01), ("novel_count", 20), ("learning_seconds", -1.)])
def test_malformed_scientific_scalars_fail_clearly(field, value):
    rows = path()
    rows[3][field] = value
    with pytest.raises(ValueError):
        analysis.summarize_path(rows, config())


def test_last_persistent_observed_downward_crossing_and_initial_guards():
    result = analysis.crossing([16, 7, 10, 6, 5], 8, frozen_initial=20)
    assert result["status"] == "valid" and result["multiple_crossings"]
    assert result["n"] == pytest.approx(2.5)
    assert analysis.crossing([16, 7, 10, 9], 8)["n"] is None
    assert analysis.crossing([16, 7, 6], 8, frozen_initial=7)["status"] == "target_already_met_at_frozen_initialization"
    assert analysis.crossing([7, 6, 5], 8, frozen_initial=20)["status"] == "target_already_met_at_zero_novel"
    assert analysis.crossing([16, None, 7], 8)["status"] == "unsupported_observed_support"


def test_whole_replica_draws_preserve_global_sign_counts_and_named_seed():
    replicas = [0, 1, 2, 3, 4, 5]
    a = analysis.paired_draws(replicas, 50, 7)
    np.testing.assert_array_equal(a, analysis.paired_draws(replicas, 50, 7))
    assert np.all((a % 2).sum(axis=1) == 3)
    assert not np.array_equal(a, analysis.paired_draws(replicas, 50, 8))
    b = analysis.paired_draws([4, 7, 8, 9], 50, 7, [-1, -1, 1, 1])
    assert np.all(b[:, :2] < 2) and np.all(b[:, 2:] >= 2)


def test_paired_effects_common_A_score_compute_components_and_secondary_B():
    result = analysis.summarize(cohort(), config())
    assert result["cohorts"]["fresh"]["replicas"] == [0, 1, 2, 3]
    assert result["comparisons"]["E-B"]["novel_auc"]["mean"] == pytest.approx(-1.875)
    assert result["comparisons"]["interaction"]["novel_auc"]["mean"] == pytest.approx(0.)
    item = result["efficiency"]["E-A"]
    p = item["point"]
    assert p["status"] == "valid" and p["target_mse"] == 8
    assert p["n"] == pytest.approx(32 / 13)
    assert p["saved_fraction"] == pytest.approx(1 - p["n"] / 4)
    assert p["matched_time_ratio"] == pytest.approx(p["matched_learning_seconds"] / 4)
    assert p["equal100_time_ratio"] == pytest.approx(2.)
    assert p["S"] == pytest.approx(p["saved_fraction"] / p["matched_time_ratio"])
    assert item["retention_supported"]
    assert item["ci"]["S"] == pytest.approx([p["S"], p["S"]])
    assert result["efficiency"]["A-A"]["point"]["saved_fraction"] == 0
    assert result["efficiency"]["E-B"]["point"]["target_mse"] == 6
    json.dumps(result, allow_nan=False)


def test_supplemental_B_minus_A_reuses_paired_draws_and_preserves_existing_results():
    rows, cfg = cohort(range(6)), config()
    signs = [-1, -1, 1, -1, 1, 1]
    for row in rows:
        replica = row["metadata"]["replica"]
        row["metadata"]["global_sign"] = signs[replica]
        if row["metadata"]["recipe"]["arm"] == "B":
            for m in row["metrics"][1:]:
                m["novel_mse"] += replica * .1
    result = analysis.summarize(rows, cfg)
    before = deepcopy(result)
    report._add_incumbent_contrast(dict(records=rows, cfg=cfg), result)
    bootstrap = before["cohorts"]["fresh"]["bootstrap"]
    draws = analysis.paired_draws(list(range(6)), bootstrap["draws"], bootstrap["seed"], signs)
    for field in ("novel_auc", "Havg", "H100", "current_whole_rms"):
        differences = np.asarray(before["methods"]["B"][field]["per_replica"]) - np.asarray(
            before["methods"]["A"][field]["per_replica"])
        effect = result["comparisons"]["B-A"][field]
        assert effect["mean"] == pytest.approx(differences.mean())
        np.testing.assert_allclose(effect["ci"], np.quantile(differences[draws].mean(axis=1), [.025, .975]))
        assert effect["per_replica"] == differences.tolist()
    assert list(result["comparisons"])[0] == "B-A"
    assert "supplemental row" in report._conclusions(result)
    del result["comparisons"]["B-A"]
    del result["report_additions"]
    assert result == before


def test_supplemental_B_minus_A_handles_pending_and_single_replica():
    for replicas in (range(0), range(1)):
        rows, cfg = cohort(replicas), config()
        result = analysis.summarize(rows, cfg)
        report._add_incumbent_contrast(dict(records=rows, cfg=cfg), result)
        if replicas:
            effect = result["comparisons"]["B-A"]["novel_auc"]
            assert effect["mean"] == pytest.approx(-1.25)
            assert effect["ci"] is None
        else:
            assert result["comparisons"] == {}


def test_retention_roots_are_within_replicas_not_after_averaging():
    rows = cohort()
    for row in rows:
        if row["metadata"]["recipe"]["arm"] == "E":
            for m in row["metrics"][3:]:
                m["old_mse"] = 1. if row["metadata"]["replica"] % 2 else 16.
    result = analysis.summarize(rows, config())
    assert result["comparisons"]["E-A"]["H100"]["mean"] == pytest.approx(.5)
    assert not result["efficiency"]["E-A"]["retention_supported"]


def test_partial_blocks_visible_but_excluded_and_stationary_has_no_S_or_intervals():
    rows = cohort()[:-1] + cohort(range(2), "stationary")
    rows.append({**record("E", 3), "status": "incomplete", "metrics": []})
    result = analysis.summarize(rows, config())
    fresh = result["cohorts"]["fresh"]
    assert fresh["replicas"] == [0, 1, 2]
    assert fresh["partial_blocks"] == [dict(replica=3, complete_arms=list("ABCD"))]
    assert result["incomplete"] == [dict(cohort="fresh", replica=3, arm="E")]
    stationary = result["cohorts"]["stationary"]
    assert stationary["status"] == "diagnostic" and stationary["efficiency"] == {}
    assert stationary["methods"]["A"]["current_whole_rms"]["ci"] is None
    assert stationary["methods"]["A"]["final_novel_mse"]["per_replica"] == [8.1, 8.1]


@pytest.mark.parametrize("corruption", ["hash", "initial", "clock", "duplicate", "recipe", "short"])
def test_unpaired_or_incompatible_complete_evidence_rejected(corruption):
    rows = cohort()
    row = rows[-1]
    if corruption == "hash":
        row["metadata"]["pairing_asset_hash"] = "different"
    elif corruption == "initial":
        row["metrics"][0]["old_mse"] = 5
    elif corruption == "clock":
        row["metrics"][-1]["novel_count"] = 6
    elif corruption == "duplicate":
        rows.append(deepcopy(row))
    elif corruption == "recipe":
        row["metadata"]["recipe"]["ridge"] = 10
    elif corruption == "short":
        row["metrics"] = row["metrics"][:-1]
    with pytest.raises(ValueError):
        analysis.summarize(rows, config())


def test_sealed_summary_is_checked_against_accepted_rows_and_selection():
    rows = cohort()
    rows[0]["summary"] = analysis.summarize_path(rows[0]["metrics"], config())
    selected = {"recipes": {r["metadata"]["recipe"]["arm"]: r["metadata"]["recipe"] for r in rows[:5]}}
    analysis.summarize(rows, config(), selected)
    rows[0]["summary"]["novel_auc"] += 1
    with pytest.raises(ValueError, match="summary disagrees"):
        analysis.summarize(rows, config(), selected)
    rows[0].pop("summary")
    selected["recipes"]["A"] = {**selected["recipes"]["A"], "ridge": 10}
    with pytest.raises(ValueError, match="frozen selection"):
        analysis.summarize(rows, config(), selected)


def test_frozen_allocation_and_named_bootstrap_seed_override_config_defaults():
    rows = cohort(range(2)) + cohort(range(2), "stationary")
    selected = {"recipes": {r["metadata"]["recipe"]["arm"]: r["metadata"]["recipe"] for r in rows[:5]},
                "main_replicas": 2, "stationary_replicas": 2, "bootstrap_seed": 724}
    result = analysis.summarize(rows, config(), selected)
    assert result["cohorts"]["fresh"]["planned_pairs"] == 2
    assert result["cohorts"]["fresh"]["bootstrap"]["seed"] == 724
    assert result["status"] == "completed"


def test_completed_report_distinguishes_unsupported_results_from_pending_work():
    rows = cohort() + cohort(range(2), "stationary")
    for row in rows:
        if row["metadata"]["cohort"] == "fresh" and row["metadata"]["recipe"]["arm"] == "E":
            row["metrics"] = path("E", end=12.)
    result = analysis.summarize(rows, config())
    assert result["status"] == "completed"
    text = report._conclusions(result)
    assert "not estimable" in text and "pending" not in text
    assert "STM-only B has an observed" in text
    assert "assess retention separately" in text
    progress = report._progress(dict(status={"status": "completed"}, records=rows, cfg=config()), result)
    assert "**Progress: Complete**" in progress


def test_failed_inverse_bootstrap_withholds_all_inverse_intervals_not_effects():
    rows = cohort()
    # Mean E endpoint is better than A. Some whole-pair draws nevertheless
    # include enough high-error replicas to leave its crossing unsupported.
    for row in rows:
        if row["metadata"]["recipe"]["arm"] == "E" and row["metadata"]["replica"] in (0, 1):
            row["metrics"] = path("E", end=12.)
    result = analysis.summarize(rows, config())
    item = result["efficiency"]["E-A"]
    assert item["point"]["status"] == "valid"
    assert item["bootstrap_crossing_failures"] > 0
    assert item["ci"] is None
    assert item["inverse_interval_status"].startswith("withheld")
    assert result["comparisons"]["E-A"]["novel_auc"]["ci"] is not None


def test_weak_reference_target_is_labeled_not_changed():
    rows = cohort()
    for row in rows:
        if row["metadata"]["recipe"]["arm"] == "A":
            row["metrics"] = path("A", end=15.5)
    result = analysis.summarize(rows, config())
    item = result["efficiency"]["E-A"]
    assert item["point"]["weak_target"]
    assert item["point"]["target_mse"] == 15.5
    assert not item["sample_saving_supported"]


def test_completed_low_novel_count_seed_is_retained_without_inverse_or_auc():
    rows = cohort()
    for row in rows:
        if row["metadata"]["replica"] == 0:
            row["metrics"] = [m for m in row["metrics"] if m["accepted_step"] != 4]
            row["metrics"][-1]["novel_count"] = 2
    result = analysis.summarize(rows, config())
    assert result["cohorts"]["fresh"]["replicas"] == [0, 1, 2, 3]
    assert result["methods"]["A"]["novel_auc"]["status"] == "unsupported"
    assert result["efficiency"]["E-A"]["ci"] is None
    assert result["efficiency"]["E-A"]["point"]["S"] is None
    assert result["methods"]["A"]["current_whole_rms"]["ci"] is not None
    json.dumps(result, allow_nan=False)


def _scientific_store(root, rows):
    from mnist_experiment.fisher_memory.artifacts import Store

    with Store(root, {"protocol": analysis.PROTOCOL}, max_bytes=64_000_000,
               free_reserve_bytes=0, finalization_reserve_bytes=16_384) as store:
        for row in rows:
            uid = row["unit_id"]
            store.begin(uid, row["metadata"])
            if row["status"] == "completed":
                store.write_json(uid, "summary.json", analysis.summarize_path(row["metrics"], config()))
                store.write_json(uid, "metrics.json", row["metrics"])
                store.write_json(uid, "diagnostics.json", row["diagnostics"])
                store.complete(uid)


def _tree_hash(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and "report" not in p.relative_to(root).parts}


def test_pending_publication_without_root_is_valid_embedded_and_idempotent(tmp_path):
    root, notebook = tmp_path / "not-yet-created", tmp_path / "public.ipynb"
    first = report.build_report(root, cfg=config(), notebook_path=notebook)
    assert first["changed"] and first["figure_count"] == 5
    assert not (root / "identity.json").exists() and not (root / "units").exists()
    book = nbformat.read(notebook, as_version=4)
    nbformat.validate(book)
    assert "Progress:" in book.cells[0].source and "execution pending" in book.cells[0].source
    assert str(root.resolve()) not in notebook.read_text()
    assert len([c for c in book.cells if c.get("attachments")]) == 5
    assert all(c.cell_type == "markdown" for c in book.cells)
    assert sum(line.startswith("| --- |") for c in book.cells for line in c.source.splitlines()) == 3
    before = notebook.read_bytes()
    second = report.build_report(root, cfg=config(), notebook_path=notebook)
    assert not second["changed"] and second["generation"] == first["generation"]
    assert notebook.read_bytes() == before
    forced = report.build_report(root, cfg=config(), notebook_path=notebook, force=True)
    assert forced["generation"] != first["generation"]
    assert (root / "report" / first["generation"] / "COMPLETED").exists()
    reused = report.build_report(root, cfg=config(), notebook_path=notebook)
    assert not reused["changed"] and reused["generation"] == forced["generation"]


def test_completed_science_is_read_only_and_status_change_creates_new_generation(tmp_path):
    root, notebook = tmp_path / "run", tmp_path / "results.ipynb"
    rows = cohort(range(2)) + cohort(range(2), "stationary") + cohort(range(1), "development")
    _scientific_store(root, rows)
    before = _tree_hash(root)
    first = report.build_report(root, cfg=config(), notebook_path=notebook)
    assert first["completed_pairs"] == {"fresh": 2, "stationary": 2}
    assert _tree_hash(root) == before
    original = (root / "report" / first["generation"] / "results.ipynb").read_bytes()
    assert "| B-A |" in original.decode()
    exported = json.loads((root / "report" / first["generation"] / "analysis.json").read_text())
    assert exported["comparisons"]["B-A"]["novel_auc"]["mean"] == pytest.approx(-1.25)
    status = dict(phase="development", active_seconds=12, current=dict(unit="a-unit", batch=3),
                  completed=dict(development=5, fresh=10, stationary=10), forecast_remaining_seconds=3600,
                  resources=dict(filesystem_free_bytes=50_000_000_000))
    (root / "status.json").write_text(json.dumps(status))
    (root / "readiness.json").write_text(json.dumps(dict(rendered_grid=np.zeros((120, 240)).tolist())))
    second = report.build_report(root, cfg=config(), notebook_path=notebook)
    assert second["changed"] and second["generation"] != first["generation"]
    assert (root / "report" / first["generation"] / "results.ipynb").read_bytes() == original
    assert "a-unit / 3" in nbformat.read(notebook, as_version=4).cells[0].source
    assert "1.00 active h" in nbformat.read(notebook, as_version=4).cells[0].source


def test_report_cancellation_does_not_publish_partial_generation(tmp_path):
    root, notebook = tmp_path / "run", tmp_path / "results.ipynb"
    first = report.build_report(root, cfg=config(), notebook_path=notebook)
    previous = (root / "report" / "latest.json").read_bytes()
    old_notebook = notebook.read_bytes()
    ticks = 0

    def check():
        nonlocal ticks
        ticks += 1
        if ticks == 15:
            raise InterruptedError("graceful pause")

    with pytest.raises(InterruptedError):
        report.build_report(root, cfg=config(), notebook_path=notebook, force=True, check=check)
    assert (root / "report" / "latest.json").read_bytes() == previous
    assert notebook.read_bytes() == old_notebook
    assert (root / "report" / first["generation"] / "COMPLETED").exists()
    assert not list((root / "report").glob(".tmp-generation-*"))


def test_corrupt_completed_science_fails_without_repair_or_republication(tmp_path):
    root, notebook = tmp_path / "run", tmp_path / "results.ipynb"
    _scientific_store(root, cohort(range(1)))
    report.build_report(root, cfg=config(), notebook_path=notebook)
    previous = (root / "report" / "latest.json").read_bytes()
    metrics = root / "units" / "fresh-0-A" / "metrics.json"
    metrics.write_text("[]")
    with pytest.raises(RuntimeError, match="changed|mismatch"):
        report.build_report(root, cfg=config(), notebook_path=notebook)
    assert metrics.read_text() == "[]"
    assert (root / "report" / "latest.json").read_bytes() == previous


def test_tampered_report_generation_is_not_overwritten(tmp_path):
    root, notebook = tmp_path / "run", tmp_path / "results.ipynb"
    generation = report.build_report(root, cfg=config(), notebook_path=notebook)["generation"]
    image = root / "report" / generation / "figures" / "health.png"
    image.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="generation changed"):
        report.build_report(root, cfg=config(), notebook_path=notebook)
    assert image.read_bytes() == b"not an image"
