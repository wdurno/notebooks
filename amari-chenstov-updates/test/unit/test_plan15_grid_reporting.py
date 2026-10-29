"""Small scalar fixtures exercise Phase 11 reporting, without models or data."""

import base64
from contextlib import contextmanager
from copy import deepcopy
import io
import json

import nbformat
import numpy as np
from PIL import Image
import pytest

from mnist_experiment.fisher_memory_grid_registration import report


def pending():
    return dict(schema_version=1, protocol=report.PROTOCOL,
        config=dict(init_size=3000, novel_count=100, retention_margin=.5, matching_tolerance=.1, envelope_factor=2.5),
        progress=dict(status="running", stage="11A", active_seconds=5),
        decisions=dict(initialization={}, readiness={}, selection={}, manifest={}),
        development={"grid": []}, analysis={"grid": None})


def analyzed(N=100):
    methods = {}
    for index, method in enumerate(report.LABELS):
        risk = np.linspace(16, 2 + index, N+1)
        old = np.linspace(4, 4.4, N+1)
        methods[method] = dict(acquisition=dict(mean=6. + index, ci=[5.+index, 7.+index]),
            retention=dict(mean=.2, ci=[.1, .3], upper_one_sided_95=.28),
            costs=dict(mean_learning_seconds=300. if method == report.FD else 100.),
            curves=dict(n=list(range(N+1)), learning_seconds=np.linspace(0, 300 if method == report.FD else 100, N+1).tolist(),
                novel=dict(mean_mse=risk.tolist(), mean_rmse=[999.]*(N+1),
                    ci_mse=[(risk*.9).tolist(), (risk*1.1).tolist()], mean_mae=(np.sqrt(risk)*.8).tolist(), mean_accuracy1=[.3]*(N+1)),
                base=dict(mean_mse=old.tolist(), mean_rmse=[999.]*(N+1),
                    ci_mse=[(old*.9).tolist(), (old*1.1).tolist()], mean_mae=(np.sqrt(old)*.8).tolist(), mean_accuracy1=[.4]*(N+1))))
    point = dict(status="valid", matching_budget=.85209*N, target_mse=3., replay_target_rmse=np.sqrt(3.),
        saved_fraction=.147908, learning_time_ratio=2.077731, S=.071187, useful_target=True,
        fd_old_rmse=2.1, replay_old_rmse=2.05, old_rmse_difference=.05)
    ci = dict(S=[.05, .09], saved_fraction=[.1, .2], matching_budget=[.8*N, .9*N],
              learning_time_ratio=[2., 2.2], old_rmse_difference=[.01, .08])
    return dict(protocol=report.PROTOCOL, schema_version=1, context_key="grid", output_dim=54,
        status="completed", is_final=True, replicas=[0, 1], N=N, methods=methods,
        units=dict(risk="square_pixels_per_landmark"),
        G=dict(mean=1., ci=[.1, 2.]), comparisons={f"{report.FD}_vs_{report.REPLAY}": dict(
            retention_difference=dict(mean=.05, ci=[.01, .08], upper_one_sided_95=.075))},
        efficiency=dict(point=point, bootstrap=dict(ci=ci, invalid=0, draws=2000, multiple_crossing_draws=3),
            full_budget_retention_supported=True, matched_budget_retention_supported=True,
            retention_qualified=True, evidence_sample_savings=True, equal_budget_learning_time_ratio=99.))


def complete():
    snapshot = pending()
    snapshot["analysis"]["grid"] = analyzed()
    snapshot["preview"] = dict(reference=np.eye(84).tolist(), query=np.eye(84).tolist(),
        true_landmarks=[[[-6, -4], [6, -4], [0, 8]]]*9,
        predicted_landmarks=[[[-5, -4], [6, -4], [0, 7]]]*9)
    snapshot["schedule"] = dict(group=[1, 40, 200], p=[0, .1, .1], u=[.5, .55, 1.])
    for index, method in enumerate(report.LABELS):
        row = dict(method=method, recipe_key=method, recipe=dict(ridge=1., strength=2. if method == report.FD else 0.),
                   novel_auc=6.+index, base_harm=.2, seconds_per_arrival=.6 if method == report.FD else .3,
                   resource_feasible=True)
        snapshot["development"]["grid"].append(row)
        snapshot["decisions"]["selection"].setdefault("grid", {})[method] = dict(selected=deepcopy(row), eligible=True, reason="Eligible")
    snapshot["decisions"]["initialization"]["grid"] = dict(channels=[4, 6], epochs=15,
        ratios=dict(grid=.25), capability_supported=True)
    snapshot["decisions"]["resource-assessment"] = dict(backend="cuda_exact", rows=[
        dict(name=f"kernel-{n}", cpu_seconds=.12, cuda_seconds=.02, status="Measured") for n in range(12)])
    snapshot["per_cell"] = dict(grid={m: [1., 2., 3., 4., 5., 6., 7., 8., 9.] for m in report.LABELS})
    return snapshot


def prose(cells):
    return "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in cells)


def test_progress_timestamp_and_mixed_readiness_are_visible():
    snapshot = pending()
    snapshot["progress"]["heartbeat"] = "2026-10-29T03:50:00+00:00"
    snapshot["decisions"]["readiness"] = dict(blur_max=2.4,grid={"2.4": [
        dict(frozen_degradation_ratio=1.077,recovered=False),
        dict(frozen_degradation_ratio=1.199,recovered=True)]})
    text = prose(report.build_cells(snapshot))
    assert "2026-10-29T03:50:00+00:00" in text
    assert "1/2 development replicas" in text
    assert "mixed recoverability" in text
    assert "7.7%" in text and "19.9%" in text


def test_subspace_cost_and_accuracy_use_comparable_gpu_labels():
    snapshot = pending()
    snapshot["decisions"]["resource-assessment"] = dict(selected_backend="subspace_cuda",
        backends={"subspace_cuda": dict(validated=True,median_transaction_seconds=.07)},
        numerical_comparison=dict(candidate_median_seconds=.07,exact_median_seconds=.57,
            max_matrix_relative_error=.00014,max_prediction_relative_error=.00089))
    text = prose(report.build_cells(snapshot))
    assert "Subspace 0.07 s / exact CUDA 0.57 s" in text
    assert "0.014%" in text and "0.089%" in text
    assert "tied cutoffs" in text
    assert "historical fallback totals are unavailable, not zero" in text
    assert "CPU Unavailable" not in text


def confirmation(method, replica=0, h=.2):
    from mnist_experiment.fisher_memory_grid_registration.config import Config, Recipe
    from mnist_experiment.fisher_memory_grid_registration.analysis import summarize_trajectory

    recipe = Recipe.from_config(Config.smoke_config(), method, ridge=1., strength=2. if method == report.FD else 0.)
    novel = [16., 8., 4., 2.] if method == report.FD else [16., 12., 10., 8.]
    metrics = []
    for n in range(4):
        base = (2+h)**2 if n else 4.
        risk = novel[n]*(1+replica)
        metrics.append(dict(n=n, observations=(10+replica)*n,
            learning_seconds=(2 if method == report.FD else 1)*n,
            base_mse=base, novel_mse=risk, base_mae=np.sqrt(base)*.8, novel_mae=np.sqrt(risk)*.8,
            base_accuracy1=.4, novel_accuracy1=.3,
            coordinate_mse=dict(base=[base/2]*54, novel=[risk/2]*54),
            per_cell_mse=dict(base=[base]*9, novel=[risk]*9)))
    return dict(summarize_trajectory(metrics, N=3), family="confirmation--grid", partition="confirmation",
        context_key="grid", output_dim=54, target=3, replica=replica, method=method,
        recipe=recipe.mapping(), recipe_key=recipe.key, capacity=recipe.capacity, replay_count=recipe.replay_count,
        plan_hash=f"plan-{replica}", pairing_id=f"plan-{replica}", active_seconds=3., peak_vram_bytes=0,
        metrics=metrics)


class Store:
    def __init__(self, root):
        self.root, self.reservations = root, []

    @contextmanager
    def reserve(self, size):
        self.reservations.append(size)
        yield


def test_pending_append_preserves_all_history_outputs_and_metadata_and_is_idempotent(tmp_path):
    path = tmp_path / "results.ipynb"
    history = [nbformat.v4.new_markdown_cell("# Results", id="title"),
        nbformat.v4.new_code_cell("raise RuntimeError('never execute')", id="historic-output", execution_count=18,
            outputs=[nbformat.v4.new_output("stream", name="stdout", text="original output\n")]),
        nbformat.v4.new_markdown_cell("Phase 9 exact conclusion", id="plan15-phase9-conclusion"),
        nbformat.v4.new_markdown_cell("Phase 10 exact conclusion", id="plan15-phase10-conclusion", metadata={"plan15_phase10": True})]
    history[1]["source"] = ["raise RuntimeError('never execute')\n"]
    book = dict(nbformat.v4.new_notebook(cells=history, metadata={"custom": {"nested": [1, 2]}}))
    path.write_text(json.dumps(book))
    snapshot, before = pending(), deepcopy(history)
    report.publish(snapshot, path)
    first = json.loads(path.read_bytes())
    assert [c for c in first["cells"] if not report.owned(c)] == before
    assert first["metadata"] == book["metadata"]
    assert first["cells"][1]["id"] == "plan15-phase11-progress"
    assert next(i for i, c in enumerate(first["cells"]) if c["id"] == "plan15-phase10-conclusion") < next(
        i for i, c in enumerate(first["cells"]) if c["id"] == "plan15-phase11-intro")
    assert "unverified" in first["cells"][1]["source"]
    assert all(c["cell_type"] == "markdown" and c["metadata"][report.TAG] for c in first["cells"] if report.owned(c))
    assert "Waiting for complete fresh three-arm replica cohorts" in prose(first["cells"])
    report.publish(snapshot, path)
    assert json.loads(path.read_bytes()) == first
    nbformat.validate(nbformat.read(path, as_version=4))


def test_replace_only_phase11_owned_cells():
    assert report.owned({"id": "plan15-phase11-old"})
    assert report.owned({"id": "arbitrary", "metadata": {report.TAG: True}})
    assert not report.owned({"id": "plan15-phase110-original"})
    assert not report.owned({"id": "plan15-phase10-original", "metadata": {report.TAG: False}})


def test_frozen_archive_and_oversize_input_remain_untouched(tmp_path, monkeypatch):
    path = tmp_path / "archive.ipynb"
    path.write_text(json.dumps(dict(nbformat.v4.new_notebook(
        metadata={"plan15_phase8_presentation": {"role": "frozen_archive"}}))))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="frozen"):
        report.publish(pending(), path)
    monkeypatch.setattr(report, "MAX_NOTEBOOK_BYTES", 2)
    with pytest.raises(ValueError, match="size"):
        report.publish(pending(), path)
    assert path.read_bytes() == before


def test_complete_report_has_five_nonblank_figures_three_tables_and_grid_units(tmp_path):
    snapshot = complete()
    before = deepcopy(snapshot)
    cells = report.build_cells(snapshot, "supplement.json")
    figures = [c for c in cells if c.get("attachments")]
    assert len(figures) == 5
    assert sum("| ---" in c["source"] for c in cells) == 3
    for figure in figures:
        encoded = next(iter(figure["attachments"].values()))["image/png"]
        pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))
        assert pixels.shape[0] > 200 and pixels.shape[1] > 600 and pixels.std() > 10
    text = prose(cells)
    for phrase in ("nine digit pairs", "100 novel grids = 900 novel digit pairs", "27 landmarks",
                   "4I_{54}", "J_i^TJ_i/4", "not 54 coordinates", "Both new and old panels are held out",
                   "3,268 LTM + 344 STM = 3,612", "CPU 0.12 s / CUDA 0.02 s", "**$S$**",
                   "STM-to-LTM", "retired-training-cohort", "Machine-readable supplement"):
        assert phrase in text
    assert "Profile: kernel-7" in text and "Profile: kernel-8" not in text
    assert snapshot == before
    path = tmp_path / "bounded.ipynb"
    report.publish(snapshot, path)
    nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.parametrize("mutation", [dict(context_key="vector"), dict(output_dim=6),
    dict(protocol="plan15-registration-v1"), dict(schema_version=2), dict(status="running"),
    dict(methods={}), dict(replicas=[0])])
def test_incompatible_analysis_fails_clearly_without_relabeling_or_repair(mutation):
    snapshot = pending()
    snapshot["analysis"]["grid"] = {**analyzed(), **mutation}
    with pytest.raises(ValueError, match="compatible"):
        report.build_cells(snapshot)
    snapshot["analysis"] = dict(vector=analyzed())
    with pytest.raises(ValueError, match="keyed by grid"):
        report.build_cells(snapshot)


def test_pending_and_partial_confirmation_never_call_inference_or_models(monkeypatch):
    import torch
    from mnist_experiment.fisher_memory_registration import analysis

    def forbidden(*args, **kwargs):
        raise AssertionError("report must not compute inference or load model artifacts")

    for name in ("analyze", "paired_analysis", "replica_draw_indices"):
        monkeypatch.setattr(analysis, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    snapshot = pending()
    snapshot["config"]["novel_count"] = 3
    snapshot["confirmation"] = [confirmation(report.FD)]
    text = prose(report.build_cells(snapshot))
    assert "Partial cohorts have no inference" in text
    assert "Samples saved |" not in text
    snapshot["analysis"]["grid"] = analyzed()
    assert "Paired acquisition gain" in prose(report.build_cells(snapshot))


def test_comparison_table_prominent_score_both_units_and_actual_equal_budget_cost():
    result = analyzed()
    before = deepcopy(result)
    text = report._comparison_table(result)
    assert "| **$S$** | **0.0712 [0.05, 0.09]** |" in text
    assert "14.79% [10.00%, 20.00%]" in text
    assert "85.21 [80.00, 90.00] / 100" in text
    assert "766.88 [720.00, 810.00] / 900" in text
    assert "| Matched-quality compute (FD / replay) | 2.08x [2x, 2.2x] |" in text
    assert "| Equal-budget compute (FD / replay) | 3x |" in text
    assert "0.05 [0.01, 0.08]" in text
    assert result == before


def test_unresolved_inverse_withholds_all_inverse_intervals_but_not_full_budget_ci():
    snapshot = pending()
    result = analyzed()
    result["efficiency"]["bootstrap"].update(invalid=1, ci=None)
    result["efficiency"].update(retention_qualified=False, evidence_sample_savings=False)
    snapshot["analysis"]["grid"] = result
    text = prose(report.build_cells(snapshot))
    assert "0.0712 (descriptive; inverse CI withheld)" in text
    assert "1/2000 unresolved draws" in text
    assert "Full-budget retention supported; matched-budget unresolved" in text
    assert "| Matched-budget retention | Unresolved |" in text
    assert "no failed inversion was discarded" in text and "unresolved is not zero" in text
    assert "retention-unqualified" in text and "not establish a practical efficiency win" in text
    assert "Paired acquisition gain (replay minus FD): 1 [0.1, 2]" in text
    assert "H difference (FD minus replay): 0.05 [0.01, 0.08]" in text
    # Even an inconsistent cached interval must not resurrect successful-only CIs.
    result["efficiency"]["bootstrap"]["ci"] = analyzed()["efficiency"]["bootstrap"]["ci"]
    assert "[0.05, 0.09]" not in report._comparison_table(result)


@pytest.mark.parametrize("status", ["target_not_reached_at_observed_endpoint", "target_already_met_at_initialization"])
def test_failed_point_is_unresolved_not_zero_and_weak_target_is_prominent(status):
    snapshot = pending()
    result = analyzed()
    result["efficiency"]["point"] = dict(status=status, useful_target=False, target_mse=3.)
    result["efficiency"]["bootstrap"].update(ci=None, invalid=8)
    result["efficiency"].update(retention_qualified=False, evidence_sample_savings=False)
    snapshot["analysis"]["grid"] = result
    text = prose(report.build_cells(snapshot))
    assert "| Samples saved | Unresolved |" in text
    assert "| **$S$** | **Unresolved** |" in text
    assert status.replace("_", " ") in text
    assert "**Weak target:**" in text and "less than 5%" in text
    assert "Full-budget retention supported; matched-budget unresolved" in text


def test_nonfinal_smoke_is_not_presented_as_fresh_confirmation():
    snapshot = pending()
    result = analyzed()
    result.update(status="smoke_completed", is_final=False)
    result["efficiency"]["evidence_sample_savings"] = False
    snapshot["analysis"]["grid"] = result
    assert "Smoke/development evidence, not final confirmation" in prose(report.build_cells(snapshot))


def test_risk_plots_root_mean_risk_and_show_both_frozen_domains(monkeypatch):
    figures = []
    monkeypatch.setattr(report, "picture", lambda name, figure, caption: figures.append(figure) or caption)
    result = analyzed()
    caption = report._predictive(result)
    for axis, domain in zip(figures[0].axes[:2], ("novel", "base")):
        for line, method in zip(axis.lines, report.LABELS):
            np.testing.assert_allclose(line.get_ydata(), np.sqrt(result["methods"][method]["curves"][domain]["mean_mse"]))
        assert axis.lines[-1].get_label() == "Frozen initialization"
    assert "not mean replica RMS" in caption and "27 landmarks" in caption
    result["efficiency"]["bootstrap"]["ci"] = None
    report._efficiency(result)
    quality = figures[1].axes[0]
    np.testing.assert_allclose(quality.lines[0].get_ydata(), np.sqrt(result["methods"][report.FD]["curves"]["novel"]["mean_mse"]))
    assert "inverse CI unresolved" in quality.get_title()
    assert quality.child_axes[0].get_xlabel() == "Novel digit pairs"


def test_frontier_uses_selected_recipe_in_grid_cohort_not_finalist_scores(monkeypatch):
    snapshot, figures = complete(), []
    for choice in snapshot["decisions"]["selection"]["grid"].values():
        choice["selected"].update(base_harm=20, novel_auc=99, seconds_per_arrival=100)
    monkeypatch.setattr(report, "picture", lambda name, figure, caption: figures.append(figure) or caption)
    caption = report._frontier(snapshot)
    stars = [c for c in figures[0].axes[0].collections if c.get_sizes()[0] == 90]
    for star, row in zip(stars, snapshot["development"]["grid"]):
        np.testing.assert_allclose(star.get_offsets()[0], [row["base_harm"], row["novel_auc"]])
    assert [t.get_text() for t in figures[0].axes[1].texts] == ["2.0x"]
    assert "same cohort" in caption and "expanded-cohort" in caption
    snapshot["decisions"]["selection"] = {}
    report._frontier(snapshot)
    assert [t.get_text() for t in figures[-1].axes[1].texts] == ["pending"]


def test_preview_offsets_are_row_major_and_never_modify_cell_local_coordinates(monkeypatch):
    snapshot, figures = complete(), []
    before = deepcopy(snapshot)
    monkeypatch.setattr(report, "picture", lambda name, figure, caption: figures.append(figure) or caption)
    report._overview(snapshot)
    true = np.asarray(figures[0].axes[0].collections[0].get_offsets())
    np.testing.assert_allclose(true[0], [84+13.5-6, 13.5-4])
    np.testing.assert_allclose(true[-3], [84+56+13.5-6, 56+13.5-4])
    assert snapshot == before
    snapshot["preview"]["true_landmarks"] = [[0, 0]]*3
    with pytest.raises(ValueError, match="cell-local"):
        report._overview(snapshot)
    snapshot["preview"]["reference"] = np.eye(28).tolist()
    with pytest.raises(ValueError, match="84x84"):
        report._overview(snapshot)


def test_cell_panel_never_silently_reshapes_or_bootstraps_cells(monkeypatch):
    snapshot = complete()
    monkeypatch.setattr(report, "picture", lambda name, figure, caption: caption)
    snapshot["per_cell"]["grid"][report.FD] = [1.]*8
    with pytest.raises(ValueError, match="nine nonnegative"):
        report._applied(snapshot, analyzed())
    snapshot.pop("per_cell")
    assert "not independent replica" in report._applied(snapshot, analyzed())


def test_content_addressed_json_preserves_full_diagnostics_and_is_not_overwritten(tmp_path):
    store = Store(tmp_path / "cache")
    snapshot = pending()
    snapshot["decisions"]["readiness"]["grid"] = dict(per_cell=[1.]*9, jacobian_spectra=[1., .5, 0.])
    snapshot["numerical_audits"] = dict(failed_inversions=[dict(draw=19, status="unresolved")])
    before = deepcopy(snapshot)
    path = tmp_path / "main.ipynb"
    report.publish(snapshot, path, store)
    supplements = list((store.root / "reports").glob("phase11-supplement-*.json"))
    assert len(supplements) == 1 and json.loads(supplements[0].read_bytes()) == snapshot
    assert len(store.reservations) == 2
    original = supplements[0].read_bytes()
    report.publish(snapshot, path, store)
    assert len(list((store.root / "reports").glob("*.json"))) == 1
    assert supplements[0].read_bytes() == original and len(store.reservations) == 3
    assert snapshot == before
    snapshot["progress"]["active_seconds"] += 1
    report.publish(snapshot, path, store)
    assert len(list((store.root / "reports").glob("*.json"))) == 2
    assert supplements[0].read_bytes() == original


def test_output_size_and_store_budget_rejections_do_not_mutate_historical_notebook(tmp_path, monkeypatch):
    path = tmp_path / "results.ipynb"
    path.write_text(json.dumps(dict(nbformat.v4.new_notebook())))
    before = path.read_bytes()
    store = Store(tmp_path / "cache")
    with monkeypatch.context() as patch:
        patch.setattr(report, "MAX_NOTEBOOK_BYTES", len(before)+1)
        with pytest.raises(ValueError, match="size"):
            report.publish(pending(), path, store)
    assert not store.reservations and not store.root.exists()

    @contextmanager
    def denied(size):
        raise RuntimeError("storage cap")
        yield

    monkeypatch.setattr(store, "reserve", denied)
    with pytest.raises(RuntimeError, match="storage cap"):
        report.publish(pending(), path, store)
    assert path.read_bytes() == before and not store.root.exists()


def test_progress_reports_verified_state_pairs_resources_and_checkpoint():
    snapshot = pending()
    snapshot["progress"].update(status="stopped", stage="11D", process_verified=True,
        pairs=dict(grid=dict(completed=3, planned=12)), active_seconds=3600, allocation_seconds=72000,
        forecast_seconds=5400, storage=dict(used_bytes=2_000_000_000, external_bytes=1_000_000_000),
        last_accepted_checkpoint="confirmation-grid-r3 n=100")
    text = report._progress(snapshot)["source"]
    for phrase in ("stopped / 11D", "3/12 complete/planned three-arm", "1.00/20 active h", "19.00 h",
                   "1.50 h", "497.00 GB headroom", "confirmation-grid-r3 n=100"):
        assert phrase in text
    snapshot["progress"]["status"] = "running"
    assert "unverified" not in report._progress(snapshot)["source"]


def test_rolling_means_use_only_whole_cohorts_and_never_compute_ci(monkeypatch):
    from matplotlib.axes import Axes
    from mnist_experiment.fisher_memory_grid_registration import analysis

    def forbidden(*args, **kwargs):
        raise AssertionError("rolling report must not bootstrap, invert or draw confidence bands")

    for name in ("analyze", "paired_analysis", "replica_draw_indices", "point_equivalence"):
        monkeypatch.setattr(analysis, name, forbidden)
    monkeypatch.setattr(Axes, "fill_between", forbidden)
    snapshot = pending()
    snapshot["config"].update(novel_count=3, confirmation_replicas=12)
    snapshot["confirmation"] = [confirmation(m) for m in report.LABELS]
    snapshot["confirmation"] += [confirmation(m, replica=1, h=.8) for m in list(report.LABELS)[:2]]
    before = deepcopy(snapshot)
    result = report._analysis(snapshot)
    assert result["provisional"] and result["replicas"] == [0] and result["incomplete_replicas"] == [1]
    assert not {"bootstrap", "efficiency", "G", "comparisons"} & set(result)
    for method, item in result["methods"].items():
        arm = next(r for r in snapshot["confirmation"] if r["replica"] == 0 and r["method"] == method)
        assert item["acquisition"] == dict(mean=arm["novel_auc"])
        assert item["retention"]["mean"] == pytest.approx(.2)
        assert item["curves"]["novel"]["mean_mse"] == [m["novel_mse"] for m in arm["metrics"]]
        assert "ci_mse" not in item["curves"]["novel"]
    cells = report.build_cells(snapshot)
    text = prose(cells)
    assert "1/12 complete/planned three-arm replica pairs" in text
    assert "1 additional replica groups await all three arms" in text
    assert "descriptive mean only" in text and "no bootstrap intervals, shading or significance claims" in text
    assert "No bootstrap, confidence intervals, significance tests or sample-equivalence estimates" in text
    assert "Bands are runner-computed" not in text and "| **$S$**" not in text
    assert len([c for c in cells if c.get("attachments")]) == 2
    assert sum("| ---" in c["source"] for c in cells) == 2
    assert snapshot == before

    snapshot["confirmation"].append(confirmation(report.EWC, replica=1, h=.8))
    result = report._analysis(snapshot)
    assert result["replicas"] == [0, 1] and not result["incomplete_replicas"]
    for item in result["methods"].values():
        assert item["retention"]["mean"] == pytest.approx(.5)


def test_incomplete_and_invalid_cohorts_remain_visible_without_selective_deletion():
    snapshot = pending()
    snapshot["config"]["novel_count"] = 3
    snapshot["confirmation"] = [confirmation(m) for m in list(report.LABELS)[:2]]
    assert report._analysis(snapshot) is None
    assert any(c["id"] == report.PREFIX+"pending" for c in report.build_cells(snapshot))
    snapshot["confirmation"].append(confirmation(report.EWC))
    snapshot["confirmation"][0]["valid"] = False
    result = report._analysis(snapshot)
    assert result["replicas"] == [0] and result["invalid_replicas"] == [0]
    text = prose(report.build_cells(snapshot))
    assert "Numerical warning" in text and "not silently excluded" in text


@pytest.mark.parametrize("mutation", [lambda rows: rows.append(deepcopy(rows[0])),
    lambda rows: rows[0].update(pairing_id="unpaired"),
    lambda rows: rows[0].update(protocol="plan15-registration-v1"),
    lambda rows: rows[0].update(output_dim=6),
    lambda rows: rows[0]["metrics"][0].update(coordinate_mse=dict(base=[1.]*54, novel=[8.]*54))])
def test_rolling_validation_rejects_duplicate_unpaired_old_protocol_and_wrong_risk(mutation):
    snapshot = pending()
    snapshot["config"]["novel_count"] = 3
    snapshot["confirmation"] = [confirmation(m) for m in report.LABELS]
    mutation(snapshot["confirmation"])
    with pytest.raises(ValueError):
        report.build_cells(snapshot)


def test_per_cell_endpoint_means_are_paired_and_ignore_waiting_replica(monkeypatch):
    figures = []
    monkeypatch.setattr(report, "picture", lambda name, figure, caption: figures.append(figure) or caption)
    snapshot = pending()
    snapshot["config"]["novel_count"] = 3
    snapshot["confirmation"] = [confirmation(m) for m in report.LABELS]
    snapshot["confirmation"].append(confirmation(report.FD, replica=1))
    report._applied(snapshot, report._analysis(snapshot))
    for line, method in zip(figures[-1].axes[1].lines, report.LABELS):
        row = next(r for r in snapshot["confirmation"] if r["method"] == method and r["replica"] == 0)
        np.testing.assert_allclose(line.get_ydata(), np.sqrt(row["metrics"][-1]["per_cell_mse"]["novel"]))
    snapshot["confirmation"].pop()
    snapshot["confirmation"].pop()
    assert report._analysis(snapshot) is None


def test_parent_resource_assessment_summaries_stay_inside_settings_table():
    snapshot = pending()
    snapshot["resource_assessment"] = dict(selected_backend="exact_cpu",
        backends=dict(exact_cpu=dict(validated=True, median_transaction_seconds=.25),
                      exact_cuda=dict(validated=False, median_transaction_seconds=None)),
        attempts=[dict(backend="exact_cuda", status="timeout")],
        whole_update_profile=dict(status="pending_parent_profile"))
    text = prose(report.build_cells(snapshot))
    assert sum("| ---" in c["source"] for c in report.build_cells(snapshot)) == 1
    assert "CPU 0.25 s / CUDA pending s" in text
    assert "Resource assessment: exact_cpu" in text and "CUDA: not validated" in text
    assert "pending_parent_profile" in text and "not a full-study forecast" in text


def test_provisional_then_final_publish_replaces_owned_cells_preserving_history_end_and_counts(tmp_path):
    path = tmp_path / "results.ipynb"
    history = [nbformat.v4.new_markdown_cell("# Results", id="title"),
               nbformat.v4.new_markdown_cell("Phase 10 exact ending", id="plan15-phase10-end")]
    path.write_text(json.dumps(dict(nbformat.v4.new_notebook(cells=history))))
    snapshot = pending()
    snapshot["config"]["novel_count"] = 3
    snapshot["confirmation"] = [confirmation(m) for m in report.LABELS]
    assert report.publish(snapshot, path) is None
    first = path.read_bytes()
    report.publish(snapshot, path)
    assert path.read_bytes() == first
    rolling = json.loads(first)
    assert rolling["cells"][-1]["id"] == report.PREFIX+"provisional"
    assert len(rolling["cells"]) == len(history)+7
    assert [c for c in rolling["cells"] if not report.owned(c)] == history

    snapshot["analysis"]["grid"] = analyzed(3)
    report.publish(snapshot, path)
    final = json.loads(path.read_bytes())
    assert [c for c in final["cells"] if not report.owned(c)] == history
    assert final["cells"][-1]["id"] == report.PREFIX+"conclusion"
    assert not any(c["id"] == report.PREFIX+"provisional" for c in final["cells"])
    assert len({c["id"] for c in final["cells"]}) == len(final["cells"])
    assert len(final["cells"]) == len(history)+8
    assert sum(bool(c.get("attachments")) for c in final["cells"] if report.owned(c)) == 3


def test_missing_target_is_a_clear_validation_error_not_eager_default_keyerror():
    snapshot = pending()
    snapshot["config"]["novel_count"] = 3
    row = confirmation(report.FD)
    row.pop("target")
    snapshot["confirmation"] = [row]
    with pytest.raises(ValueError, match="malformed or incomplete trajectory"):
        report.build_cells(snapshot)
    snapshot["config"].pop("novel_count")
    with pytest.raises(ValueError, match="integer N"):
        report.build_cells(snapshot)
