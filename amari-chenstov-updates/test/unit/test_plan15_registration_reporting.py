"""Bounded artifact reports preserve historical cells, metadata and outputs."""

from contextlib import contextmanager
from copy import deepcopy
import json

import nbformat
import numpy as np
import pytest

from mnist_experiment.fisher_memory_registration import analysis, design, report
from test_plan15_registration_analysis import cfg, cohort, pair, record, METHODS


def pending():
    return dict(config=cfg().mapping(), progress=dict(status="running", stage="10A", active_seconds=5),
                decisions=dict(initialization={}, readiness={}, selection={}, manifest={}),
                development={"vector": [], "scalar": []}, analysis={})


def test_append_only_history_and_idempotent_report_with_no_historical_execution(tmp_path):
    path = tmp_path / "results.ipynb"
    history = [nbformat.v4.new_markdown_cell("# Results", id="title"),
        nbformat.v4.new_code_cell("raise RuntimeError('never execute')", id="historic-output", execution_count=18,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="original output\n")]),
        nbformat.v4.new_markdown_cell("Phase9 exact conclusion", id="plan15-phase9-conclusion")]
    book = dict(nbformat.v4.new_notebook(cells=history, metadata={"custom": {"nested": [1, 2]}}))
    path.write_text(json.dumps(book))
    snapshot, before = pending(), deepcopy(history)
    report.publish(snapshot, path)
    first = json.loads(path.read_text())
    assert [c for c in first["cells"] if not report.owned(c)] == before
    assert first["metadata"] == book["metadata"]
    assert first["cells"][1]["id"] == "plan15-phase10-progress"
    assert next(i for i, c in enumerate(first["cells"]) if c["id"] == "plan15-phase9-conclusion") < next(
        i for i, c in enumerate(first["cells"]) if c["id"] == "plan15-phase10-intro")
    assert all(c["cell_type"] == "markdown" for c in first["cells"] if report.owned(c))
    report.publish(snapshot, path)
    assert json.loads(path.read_text()) == first
    assert "unverified" in first["cells"][1]["source"]


def test_frozen_archive_and_oversize_notebook_are_untouched(tmp_path, monkeypatch):
    path = tmp_path / "archive.ipynb"
    book = nbformat.v4.new_notebook(metadata={"plan15_phase8_presentation": {"role": "frozen_archive"}})
    path.write_text(json.dumps(book))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="frozen"):
        report.publish(pending(), path)
    assert path.read_bytes() == before
    monkeypatch.setattr(report, "MAX_NOTEBOOK_BYTES", 2)
    with pytest.raises(ValueError, match="size"):
        report.publish(pending(), path)
    assert path.read_bytes() == before


def test_two_tasks_have_at_most_five_figures_three_tables_and_distinct_pixel_axes(tmp_path):
    snapshot = pending()
    snapshot["preview"] = dict(reference=np.eye(28).tolist(), query=np.eye(28).tolist(),
        true_landmarks=[[-6, -4], [6, -4], [0, 8]], predicted_landmarks=[[-5, -4], [6, -4], [0, 7]])
    snapshot["schedule"] = dict(group=[1, 40, 200], p=[0, .1, .1], u=[.5, .55, 1.])
    for task in ("vector", "scalar"):
        snapshot["analysis"][task] = analysis.analyze(*cohort(task), seed=7, draws=20)
        rows = design.table([r for m in METHODS for r in pair(m, task=task)], cfg())
        snapshot["development"][task] = rows
        snapshot["decisions"]["selection"][task] = design.select(rows, cfg())
        snapshot["decisions"]["initialization"][task] = dict(checks=[dict(replica=0, mse=2., zero_predictor_mse=8.)])
    before = deepcopy(snapshot)
    cells = report.build_cells(snapshot, "supplement.html")
    assert len([c for c in cells if c.get("attachments")]) == 5
    assert sum("| ---" in c["source"] for c in cells) == 3
    text = "\n".join(c["source"] for c in cells)
    assert "square pixels" in text and "three landmarks" in text
    assert "scalar" in text and "common-y1" in text
    assert "same prediction coordinate" in text and "not a causal" in text
    assert "Jacobian spectra" in text
    assert snapshot == before
    report.publish(snapshot, tmp_path / "bounded.ipynb")
    nbformat.validate(nbformat.read(tmp_path / "bounded.ipynb", as_version=4))


def test_weak_target_and_failed_inverse_are_not_a_sample_savings_claim():
    snapshot = pending()
    result = analysis.analyze(*cohort(), seed=7, draws=20)
    result["efficiency"]["point"].update(useful_target=False, target_mse_gain_fraction=.01)
    result["efficiency"]["bootstrap"].update(invalid=1, ci=None)
    result["efficiency"].update(retention_qualified=False)
    snapshot["analysis"]["vector"] = result
    text = "\n".join(c["source"] for c in report.build_cells(snapshot))
    assert "1.00%" in text and "below 5%" in text
    assert "1/20 unresolved draws" in text and "no failed inversion was discarded" in text
    assert "retention-unqualified" in text
    assert "full-budget supported; matched-budget unresolved" in text
    assert "withheld inverse uncertainty, not a failed full-budget retention test" in text
    assert "Samples saved remain a descriptive point estimate" in text
    result["efficiency"]["point"]["status"] = "target_not_reached_at_observed_endpoint"
    text = "\n".join(c["source"] for c in report.build_cells(snapshot))
    assert "full-budget supported; matched-budget unresolved" in text


def test_comparison_table_exposes_s_and_distinguishes_matched_cost_from_equal_budget():
    results = {task: analysis.analyze(*cohort(task), seed=7, draws=20) for task in ("vector", "scalar")}
    vector = results["vector"]
    vector["efficiency"]["point"].update(matching_budget=85.209, saved_fraction=.147908,
        learning_time_ratio=2.077731, S=.071187, replay_target_rmse=1.064)
    vector["efficiency"]["bootstrap"].update(invalid=1, ci=None)
    vector["methods"][analysis.FD]["costs"]["mean_learning_seconds"] = 243.4
    vector["methods"][analysis.REPLAY]["costs"]["mean_learning_seconds"] = 100.
    scalar = results["scalar"]
    scalar["efficiency"]["point"] = dict(status="target_not_reached_at_observed_endpoint",
        replay_target_rmse=.601, target_mse_gain_fraction=.1365)
    scalar["efficiency"]["bootstrap"].update(invalid=12, ci=None)
    before = deepcopy(results)
    text = report._comparison_table(results)
    assert "| Statistic | Vector | Scalar |" in text
    assert "| **$S$** | **0.0712** | Unresolved |" in text
    assert "| Samples saved | 14.79% | Unresolved |" in text
    assert "| Matched-quality compute (FD / replay) | 2.08x | Unresolved |" in text
    assert "| Equal-budget compute (FD / replay) | 2.43x |" in text
    assert "1/20 unresolved draws" in text and "12/20 unresolved draws" in text
    assert "target not reached at observed endpoint" in text
    assert "descriptive point estimates" in text and "unresolved is not zero" in text
    assert results == before


def test_comparison_table_retains_inverse_intervals_and_never_infers_provisional_values():
    result = analysis.analyze(*cohort(), seed=7, draws=20)
    point, boot = result["efficiency"]["point"], result["efficiency"]["bootstrap"]
    assert boot["ci"] is not None
    text = report._comparison_table({"vector": result, "scalar": dict(provisional=True)})
    lo, hi = boot["ci"]["S"]
    assert f"| **$S$** | **{point['S']:.3g} [{lo:.3g}, {hi:.3g}]** |" in text
    lo, hi = boot["ci"]["saved_fraction"]
    assert f"{100*point['saved_fraction']:.2f}% [{100*lo:.2f}%, {100*hi:.2f}%]" in text
    assert "Scalar" not in text
    assert report._comparison_table({"scalar": dict(provisional=True)}) == ""


@pytest.mark.parametrize("channels,vector,scalar", [([4, 6], "636 (448 LTM + 188 STM)", "476 (323 LTM + 153 STM)"),
                                                  ([6, 8], "1,000 (752 LTM + 248 STM)", "792 (587 LTM + 205 STM)")])
def test_calibration_note_reports_capacity_and_mixed_readiness_without_models(channels, vector, scalar):
    snapshot = pending()
    assert report._calibration_note(snapshot) is None
    snapshot.pop("decisions")
    snapshot["initialization"] = {task: dict(channels=channels, ratios={task: .25}) for task in ("vector", "scalar")}
    snapshot["readiness"] = dict(blur_max=1.8, vector={"1.8": [dict(recovered=False), dict(recovered=True)]},
                                 scalar=[dict(recovered=False), dict(recovered=True)])
    snapshot["geometry"] = dict(rank=8, rows=[dict(rank=8, relative_frobenius=.0083),
                                               dict(rank=16, relative_frobenius=.001)])
    before = deepcopy(snapshot)
    text = report._calibration_note(snapshot)["source"]
    assert vector in text and scalar in text
    assert "75.0%" in text and "vector 1/2, scalar 1/2" in text
    assert "0.83%" in text and "not error to a known population Fisher" in text
    assert "not a guarantee of online gains" in text
    assert snapshot == before
    snapshot["decisions"] = {key: snapshot[key] for key in ("initialization", "readiness")}
    assert report._calibration_note(snapshot)["source"] == text


def test_development_cost_labels_distinguish_pending_from_selected_replay(monkeypatch):
    from matplotlib import pyplot as plt

    snapshot = pending()
    rows = design.table([r for m in METHODS for r in pair(m)], cfg())
    snapshot["development"]["vector"] = rows
    figures = []

    def capture(name, figure, caption):
        figures.append(figure)
        return caption

    monkeypatch.setattr(report, "picture", capture)
    try:
        caption = report._frontier(snapshot)
        labels = [text.get_text() for axis in figures[-1].axes for text in axis.texts]
        assert labels and set(labels) == {"pending"}
        assert "pending until replay policy selection" in caption
        snapshot["decisions"]["selection"]["vector"] = design.select(rows, cfg())
        report._frontier(snapshot)
        labels = [text.get_text() for axis in figures[-1].axes for text in axis.texts]
        assert labels and all(text.endswith("x") for text in labels)
    finally:
        for figure in figures:
            plt.close(figure)


def test_frontier_never_mixes_grid_coordinates_with_expanded_cohort_scores(monkeypatch):
    from matplotlib import pyplot as plt

    snapshot = pending()
    rows = design.table([r for m in METHODS for r in pair(m)], cfg())
    snapshot["development"]["vector"] = rows
    selected = deepcopy(design.select(rows, cfg()))
    snapshot["decisions"]["selection"]["vector"] = selected
    for choice in selected.values():
        choice["selected"]["base_harm"] += 10.
        choice["selected"]["novel_auc"] += 20.
        choice["selected"]["seconds_per_arrival"] *= 10.
    figures = []

    def capture(name, figure, caption):
        figures.append(figure)
        return caption

    monkeypatch.setattr(report, "picture", capture)
    try:
        caption = report._frontier(snapshot)
        stars = [c for c in figures[0].axes[0].collections if c.get_sizes()[0] == 90]
        for star, method in zip(stars, report.LABELS):
            key = selected[method]["selected"]["recipe_key"]
            row = next(r for r in rows if r["recipe_key"] == key)
            assert np.asarray(star.get_offsets())[0] == pytest.approx([row["base_harm"], row["novel_auc"]])
        replay = next(r for r in rows if r["method"] == analysis.REPLAY)
        fd = next(r for r in rows if r["method"] == analysis.FD)
        labels = [t.get_text() for t in figures[0].axes[1].texts]
        assert f"{fd['seconds_per_arrival']/replay['seconds_per_arrival']:.1f}x" in labels
        assert "same grid cohort" in caption and "expanded development cohort" in caption
    finally:
        for figure in figures:
            plt.close(figure)


def test_content_addressed_supplement_contains_detailed_artifacts_and_respects_store_budget(tmp_path):
    class Store:
        root = tmp_path / "cache"
        def __init__(self):
            self.reservations = []
        @contextmanager
        def reserve(self, size):
            self.reservations.append(size)
            yield
    store = Store()
    snapshot = pending()
    snapshot["decisions"]["readiness"] = dict(jacobian_spectra=[1., .5, 0.], effective_rank=2,
        per_coordinate=[1, 2, 3, 4, 5, 6])
    report.publish(snapshot, tmp_path / "main.ipynb", store)
    supplements = list((store.root / "reports").glob("phase10-supplement-*.html"))
    assert len(supplements) == 1
    text = supplements[0].read_text()
    assert "jacobian_spectra" in text and "per_coordinate" in text
    assert len(store.reservations) == 2
    before = supplements[0].read_bytes()
    report.publish(snapshot, tmp_path / "main.ipynb", store)
    assert len(list((store.root / "reports").glob("*.html"))) == 1
    assert supplements[0].read_bytes() == before and len(store.reservations) == 3


def test_one_complete_pair_displays_descriptive_means_without_bootstrap_or_ci(monkeypatch, tmp_path):
    snapshot = pending()
    rows, _ = cohort()
    snapshot["confirmation"] = [r for r in rows if r["replica"] == 0]
    snapshot["confirmation"].append(record(METHODS[0], replica=1, partition="confirmation", a=100.))
    before = deepcopy(snapshot)

    def forbidden(*args, **kwargs):
        raise AssertionError("rolling report must not bootstrap or analyze")

    monkeypatch.setattr(analysis, "analyze", forbidden)
    monkeypatch.setattr(analysis, "paired_analysis", forbidden)
    monkeypatch.setattr(analysis, "replica_draw_indices", forbidden)
    from matplotlib.axes import Axes
    monkeypatch.setattr(Axes, "fill_between", forbidden)

    result = report._analysis(snapshot)["vector"]
    assert result["provisional"] and result["replicas"] == [0]
    assert result["incomplete_replicas"] == [1]
    assert "bootstrap" not in result and "efficiency" not in result and "G" not in result
    for method in METHODS:
        arm = next(r for r in rows if r["method"] == method and r["replica"] == 0)
        item = result["methods"][method]
        assert item["acquisition"] == {"mean": arm["novel_auc"]}
        assert item["retention"]["mean"] == pytest.approx(np.sqrt(arm["base_auc"])-np.sqrt(arm["frozen_base_mse"]))
        assert item["curves"]["novel"]["mean_mse"] == [m["novel_mse"] for m in arm["metrics"]]
        assert "ci_rmse" not in item["curves"]["novel"] and "ci" not in item["retention"]
    cells = report.build_cells(snapshot)
    text = "\n".join(c["source"] for c in cells)
    assert "1 complete three-arm vector replica pair." in text
    assert "1 complete three-arm replica pair." in text
    assert "descriptive mean only" in text and "provisional" in text
    assert "no bootstrap intervals, shading or significance claims" in text
    assert "additional replica groups await all three arms" in text
    assert "Bands are pointwise" not in text and "Positive budget-averaged acquisition evidence" not in text
    assert not any(c["id"] == "plan15-phase10-efficiency" for c in cells)
    assert report._efficiency({"vector": result}) is None
    assert snapshot == before

    path = tmp_path / "rolling.ipynb"
    history = [nbformat.v4.new_markdown_cell("# Historic results", id="title"),
        nbformat.v4.new_code_cell("never_execute()", id="historic-output", execution_count=9,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="unchanged\n")]),
        nbformat.v4.new_markdown_cell("Phase 9", id="plan15-phase9-original")]
    path.write_text(json.dumps(dict(nbformat.v4.new_notebook(cells=history))))
    report.publish(snapshot, path)
    assert [c for c in json.loads(path.read_text())["cells"] if not report.owned(c)] == history
    first = path.read_bytes()
    report.publish(snapshot, path)
    assert path.read_bytes() == first


def test_rolling_means_use_only_whole_three_arm_pairs_and_roots_within_replicas():
    snapshot = pending()
    rows, _ = cohort()
    extra = [record(m, replica=2, h=.8, novel=[16., 15., 14., 12.], partition="confirmation") for m in METHODS]
    snapshot["confirmation"] = rows + extra[:2]
    result = report._analysis(snapshot)["vector"]
    assert result["replicas"] == [0, 1] and result["incomplete_replicas"] == [2]
    snapshot["confirmation"].append(extra[2])
    result = report._analysis(snapshot)["vector"]
    assert result["replicas"] == [0, 1, 2] and not result["incomplete_replicas"]
    for method in METHODS:
        group = [r for r in snapshot["confirmation"] if r["method"] == method]
        expected = np.asarray([[m["novel_mse"] for m in r["metrics"]] for r in group]).mean(axis=0)
        np.testing.assert_allclose(result["methods"][method]["curves"]["novel"]["mean_rmse"], np.sqrt(expected))
        assert result["methods"][method]["retention"]["mean"] == pytest.approx((.2+.2+.8)/3)


def test_final_analysis_replaces_only_its_task_while_other_task_still_rolls():
    snapshot = pending()
    vector_rows, manifest = cohort()
    scalar_rows, _ = cohort("scalar")
    snapshot["confirmation"] = vector_rows + [r for r in scalar_rows if r["replica"] == 0]
    analyzed = analysis.analyze(vector_rows, manifest, seed=7, draws=20)
    snapshot["analysis"] = {"vector": analyzed, "scalar": None}
    results = report._analysis(snapshot)
    assert results["vector"] is analyzed and not results["vector"].get("provisional")
    assert results["scalar"]["provisional"] and results["scalar"]["replicas"] == [0]
    cells = report.build_cells(snapshot)
    text = "\n".join(c["source"] for c in cells)
    assert "Scalar provisional confirmation" in text and "Vector provisional confirmation" not in text
    assert "vector / 2 analyzed" in text and "scalar / 1 provisional" in text
    efficiency = next(c["source"] for c in cells if c["id"] == "plan15-phase10-efficiency")
    assert "phase10-efficiency.png" in efficiency
    assert len([c for c in cells if c.get("attachments")]) <= 5
    assert sum("| ---" in c["source"] for c in cells) <= 3
    snapshot["analysis"]["scalar"] = analysis.analyze(*cohort("scalar"), seed=7, draws=20)
    assert report._analysis(snapshot)["scalar"] is snapshot["analysis"]["scalar"]
    text = "\n".join(c["source"] for c in report.build_cells(snapshot))
    assert "Scalar provisional confirmation" not in text


@pytest.mark.parametrize("bounds,direction", [([-.25, .5], "not established"),
                                             ([.01, .24], "larger in the vector context"),
                                             ([-.24, -.01], "larger in the scalar context")])
def test_common_coordinate_contrast_is_reported_from_stored_snapshot(bounds, direction):
    snapshot = pending()
    snapshot["analysis"] = {"vector": analysis.analyze(*cohort(), seed=7, draws=20)}
    snapshot["common_coordinate_contrast"] = dict(mean=float(np.mean(bounds)), ci=bounds, secondary=True, n_pairs=2)
    text = "\n".join(c["source"] for c in report.build_cells(snapshot))
    assert "Secondary common-y1 contrast" in text
    assert "vector minus scalar replay-minus-FD" in text
    assert report.interval(snapshot["common_coordinate_contrast"]) + " square pixels" in text
    assert "nominal paired 95% CI" in text and "not full registration" in text
    assert direction in text and "output dimension is not isolated" in text


def test_efficiency_plot_uses_the_mean_risk_that_is_actually_inverted(monkeypatch):
    from matplotlib import pyplot as plt

    result = analysis.analyze(*cohort(), seed=7, draws=20)
    result["efficiency"]["bootstrap"]["ci"] = None
    for method in (analysis.FD, analysis.REPLAY):
        result["methods"][method]["curves"]["novel"]["mean_rmse"] = [999.] * (result["N"]+1)
    figures = []

    def capture(name, figure, caption):
        figures.append(figure)
        return caption

    monkeypatch.setattr(report, "picture", capture)
    try:
        caption = report._efficiency({"vector": result})
        axis = figures[0].axes[0]
        for line, method in zip(axis.lines, (analysis.FD, analysis.REPLAY)):
            assert line.get_ydata() == pytest.approx(np.sqrt(result["methods"][method]["curves"]["novel"]["mean_mse"]))
        assert "point match; CI unresolved" in axis.get_title()
        assert "square root of mean replica risk" in caption
        assert "interpolation uses unrooted risk" in caption
    finally:
        for figure in figures:
            plt.close(figure)


def test_incomplete_groups_never_appear_as_complete_pairs_and_invalid_pairs_are_not_deleted():
    snapshot = pending()
    rows, _ = cohort()
    snapshot["confirmation"] = [r for r in rows if r["method"] != METHODS[-1]]
    assert report._analysis(snapshot) == {}
    assert any(c["id"] == "plan15-phase10-pending" for c in report.build_cells(snapshot))
    snapshot["confirmation"] = [r for r in rows if r["replica"] == 0]
    snapshot["confirmation"][0]["valid"] = False
    result = report._analysis(snapshot)["vector"]
    assert result["replicas"] == [0] and result["invalid_replicas"] == [0]
    text = "\n".join(c["source"] for c in report.build_cells(snapshot))
    assert "Numerical warning" in text and "not silently excluded" in text


def test_rolling_report_rejects_duplicate_unpaired_or_old_protocol_evidence():
    snapshot = pending()
    rows, _ = cohort()
    snapshot["confirmation"] = [r for r in rows if r["replica"] == 0]
    for mutation in (lambda records: records.append(deepcopy(records[0])),
                     lambda records: records[0].update(pairing_id="unpaired"),
                     lambda records: records[0].update(protocol="plan15-shift-v1")):
        damaged = deepcopy(snapshot)
        mutation(damaged["confirmation"])
        with pytest.raises(ValueError):
            report.build_cells(damaged)


def test_supplement_reads_only_sealed_geometry_and_preserves_rank_decision(tmp_path):
    class Store:
        root = tmp_path / "cache"
        def __init__(self):
            self.reads = []
        @contextmanager
        def reserve(self, size):
            yield
        def is_complete(self, unit):
            return unit == "decision-geometry-vector"
        def unit_path(self, unit):
            self.reads.append(unit)
            assert self.is_complete(unit), "incomplete geometry must not be read"
            return self.root / "units" / unit

    store = Store()
    path = store.root / "units" / "decision-geometry-vector" / "result.json"
    path.parent.mkdir(parents=True)
    spectrum = dict(eigenvalues=[0., .5, 2.], trace=2.5, effective_rank=25/17, inputs=8,
                    meaning="STM principal block of the same model Fisher")
    path.write_text(json.dumps(spectrum))
    sealed = path.read_bytes()
    snapshot = pending()
    snapshot["geometry"] = dict(rank=16, adequate=True,
        rows=[dict(task="vector", rank=16, relative_frobenius=.04)])
    before = deepcopy(snapshot)
    report.publish(snapshot, tmp_path / "results.ipynb", store)
    supplement = next((store.root / "reports").glob("phase10-supplement-*.html")).read_text()
    import html
    data = json.loads(html.unescape(supplement.split("<pre>", 1)[1].split("</pre>", 1)[0]))
    assert data["geometry"] == snapshot["geometry"]
    assert data["sealed_geometry"]["vector"]["result"] == spectrum
    assert data["sealed_geometry"]["vector"]["status"] == "completed"
    assert data["sealed_geometry"]["scalar"] == dict(unit_id="decision-geometry-scalar", status="pending")
    assert store.reads == ["decision-geometry-vector"]
    assert snapshot == before and path.read_bytes() == sealed
    report.publish(snapshot, tmp_path / "results.ipynb", store)
    assert len(list((store.root / "reports").glob("phase10-supplement-*.html"))) == 1


def test_unavailable_optional_geometry_is_reported_without_computation_or_repair(tmp_path):
    class Store:
        root = tmp_path
        def is_complete(self, unit):
            return True
        def unit_path(self, unit):
            return self.root / "units" / unit

    store = Store()
    scalar = store.unit_path("decision-geometry-scalar") / "result.json"
    scalar.parent.mkdir(parents=True)
    scalar.write_text("not yet readable JSON")
    before = scalar.read_bytes()
    geometry = report._geometry_supplement(store)
    assert all(item["status"] == "unavailable" and item["warning"] for item in geometry.values())
    assert not store.unit_path("decision-geometry-vector").exists()
    assert scalar.read_bytes() == before
