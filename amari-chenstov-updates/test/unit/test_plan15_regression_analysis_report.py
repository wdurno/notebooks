"""Synthetic scalar artifacts only; no training, evaluation, MNIST, or CUDA."""

from contextlib import contextmanager
from copy import deepcopy
import json

import nbformat
import numpy as np
import pytest

from mnist_experiment.fisher_memory_regression import analysis, report


def cohort():
    recipes = [dict(recipe_key=method, method=method, capacity=0 if method == "ewc_only" else 16,
                    replay_count=0 if method == "ewc_only" else 4)
               for method in (analysis.FD, analysis.REPLAY, "ewc_only")]
    manifest = dict(schema_version=1, protocol=analysis.PROTOCOL, sealed=True, recipes=recipes,
                    replicas=list(range(4)), N=4, n_F=2, q_dev=40., fit=False,
                    base_retention_guard_degrees=1., learning_seconds_per_observation_limit=.4)
    records = []
    for recipe in recipes:
        for replica in manifest["replicas"]:
            risks = ([100, 60, 30, 20, 10] if recipe["method"] == analysis.FD else
                     [100, 80, 60, 50, 40] if recipe["method"] == analysis.REPLAY else
                     [100, 90, 80, 70, 60])
            metrics = [dict(n=n, observations=4 * n, novel_mse=value + replica * (n + 1),
                            base_mse=9. + replica + n * .1,
                            learning_seconds=float(n * (2 if recipe["method"] == analysis.FD else 1)))
                       for n, value in enumerate(risks)]
            summary = analysis.summarize_trajectory(metrics, **recipe, replica=replica,
                                                     pairing_id=f"paired-assets-{replica}")
            records.append(dict(**summary, partition="confirmation", metrics=metrics))
    return records, manifest


def snapshot(*, complete=False):
    value = dict(schema_version=1, protocol=analysis.PROTOCOL, smoke=True, status="pending", stage="8A",
                 progress=dict(completed=0, planned=12, active_seconds=12., allocation_seconds=72000.,
                               forecast_seconds=None, storage=dict(used_bytes=1234, reserved_bytes=50)))
    if complete:
        records, manifest = cohort()
        result = analysis.analyze_confirmation(records, manifest, seed=17, draws=40)
        value.update(smoke=False, status="completed", stage="8E", analysis=result,
                     confirmation=dict(**manifest, rows=records),
                     calibration=dict(ridge=1., strength=2., fd_step=.01, cost_ratio=2.,
                                      learning_seconds_per_observation_limit=.4),
                     allocation=[{**record, "capacity": 16, "replay_count": 4} for record in records
                                 if record["method"] in (analysis.FD, analysis.REPLAY)],
                     allocation_grid=dict(capacities=[4, 16], replay_counts=[4, 16]))
    return value


def historical_notebook():
    return dict(nbformat=4, nbformat_minor=5, metadata={"arbitrary": {"unchanged": [1, 2]}}, cells=[
        dict(cell_type="markdown", id="old-title", metadata={}, source=["# Earlier work\n", "Introduction"]),
        dict(cell_type="code", id="old-code", metadata={"tags": ["plan15_phase7e"]},
             source=["print('old')\n"], execution_count=9,
             outputs=[dict(output_type="stream", name="stdout", text=["unchanged output\n"])]),
        dict(cell_type="markdown", id="old-conclusion", metadata={"nested": {"keep": True}},
             source="## Phase 7E conclusion"),
    ])


def test_budget_average_excludes_initialization_and_uses_novel_count():
    records, _ = cohort()
    record = records[0]
    assert record["novel_auc"] == pytest.approx((60 + 30 + 20 + 10) / 4)
    assert record["base_auc"] == pytest.approx(np.mean([9.1, 9.2, 9.3, 9.4]))
    assert record["observations"] == 16
    assert record["N"] == 4
    without_initial = analysis.summarize_trajectory(record["metrics"][1:], N=4,
                                                    frozen_novel_mse=100., frozen_base_mse=9.)
    assert without_initial["novel_auc"] == record["novel_auc"]


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.pop(2),
    lambda rows: rows.insert(2, dict(rows[2])),
    lambda rows: rows[1].update(n=1.2),
    lambda rows: rows[2].update(learning_seconds=0),
    lambda rows: rows[2].update(observations=3),
    lambda rows: rows[3].update(novel_mse=float("nan")),
    lambda rows: rows[0].update(observations=1),
])
def test_malformed_trajectory_rejected(mutation):
    records, _ = cohort()
    rows = deepcopy(records[0]["metrics"])
    mutation(rows)
    with pytest.raises(ValueError):
        analysis.summarize_trajectory(rows, N=4)


def test_frozen_reference_must_match_initialization():
    records, _ = cohort()
    with pytest.raises(ValueError, match="frozen risk"):
        analysis.summarize_trajectory(records[0]["metrics"], frozen_base_mse=999.)


def test_bootstrap_is_deterministic_whole_replica_and_stop_aware():
    values = np.array([1., -2., 3., 8.])
    actual = analysis.paired_interval(values, seed=29, draws=100)
    means = [values[analysis.replica_draw_indices(4, 29, draw)].mean() for draw in range(100)]
    assert actual["ci"] == pytest.approx(np.quantile(means, [.025, .975]))
    assert actual == analysis.paired_interval(values, seed=29, draws=100)
    assert actual["favorable_replicas"] == 3
    assert actual["mean"] == values.mean()
    assert actual["resampling_unit"] == "paired_whole_replica"
    with pytest.raises(InterruptedError):
        analysis.paired_interval(values, 29, check_stop=lambda: True)
    with pytest.raises(ValueError):
        analysis.paired_interval([1.], 29)


def test_confirmation_direct_counts_retention_resources_and_curve_bands():
    records, manifest = cohort()
    result = analysis.analyze_confirmation(records, manifest, 17, draws=100)
    assert result["risk_gain"]["mean"] == pytest.approx(27.5)
    assert result["direct_contrast"]["mean"] == pytest.approx(-13.)
    assert result["direct_contrast"]["n_F"] == 2
    assert result["direct_contrast"]["n_R"] == 4
    assert result["direct_contrast"]["supports_better_at_smaller_budget"]
    ret = result["retention"][analysis.FD]
    expected = np.mean([np.sqrt(9.25 + r) - np.sqrt(9. + r) for r in range(4)])
    assert ret["mean"] == pytest.approx(expected)
    assert ret["meets_guard"]
    assert result["resources"][analysis.FD]["resource_feasible"] is False
    assert result["resources"][analysis.REPLAY]["resource_feasible"] is True
    assert result["resources"][analysis.FD]["at_2"] == dict(observations=8., learning_seconds=4.)
    arrays = np.array([[row["novel_mse"] for row in record["metrics"]] for record in records[:4]])
    means = np.array([arrays[analysis.replica_draw_indices(4, 17, draw)].mean(axis=0) for draw in range(100)])
    bands = result["curves"][analysis.FD]["novel"]
    np.testing.assert_allclose(bands["ci_mse"], np.quantile(means, [.025, .975], axis=0))
    np.testing.assert_allclose(bands["mean_rmse"], np.sqrt(arrays.mean(axis=0)))
    assert result["linear_fits"] == {"registered": False}
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("defect", ["missing", "duplicate", "invalid", "development", "pairing",
                                    "summary", "recipe", "exposure", "same_replica", "unsealed"])
def test_incomplete_unpaired_or_invalid_cohort_has_no_inference(defect):
    records, manifest = cohort()
    if defect == "missing":
        records.pop()
    elif defect == "duplicate":
        records.append(deepcopy(records[0]))
    elif defect == "invalid":
        records[0]["valid"] = False
    elif defect == "development":
        records[0]["partition"] = "development"
    elif defect == "pairing":
        records[0]["pairing_id"] = "other-stream"
    elif defect == "summary":
        records[0]["novel_auc"] += 1
    elif defect == "recipe":
        records[0]["capacity"] = 32
    elif defect == "exposure":
        records[0]["metrics"][1]["observations"] = 5
    elif defect == "same_replica":
        for record in records:
            record["pairing_id"] = "same"
    else:
        manifest["sealed"] = False
    with pytest.raises(ValueError):
        analysis.analyze_confirmation(records, manifest, 17, draws=10)


def test_wrapped_artifact_records_and_analysis_cancellation():
    records, manifest = cohort()
    wrapped = [dict(metadata=dict(recipe={k: record[k] for k in ("method", "capacity", "replay_count")}),
                    summary={k: v for k, v in record.items() if k != "metrics"}, metrics=record["metrics"])
               for record in records]
    assert analysis.analyze_confirmation(wrapped, manifest, 17, 10)["cohort"]["N"] == 4
    calls = 0

    def stop():
        nonlocal calls
        calls += 1
        return calls > 20

    with pytest.raises(InterruptedError):
        analysis.analyze_confirmation(records, manifest, 17, 100, check_stop=stop)
    assert calls == 21


def test_registered_fits_are_optional_and_do_not_extrapolate():
    records, manifest = cohort()
    manifest.update(fit=True, fit_range=[1, 4], q_dev=40.)
    result = analysis.analyze_confirmation(records, manifest, 17, 10)
    assert result["linear_fits"]["registered"]
    fit = result["linear_fits"]["methods"][analysis.FD]
    assert 1 < fit["crossing"] < 4
    manifest["q_dev"] = 1000.
    invalid = analysis.analyze_confirmation(records, manifest, 17, 10)["linear_fits"]["methods"][analysis.FD]
    assert invalid["crossing"] is None
    assert "target_already_met_at_initialization" in invalid["issues"]
    assert "crossing_outside_registered_range" in invalid["issues"]
    manifest["fit_range"] = [0, 4]
    with pytest.raises(ValueError):
        analysis.analyze_confirmation(records, manifest, 17, 10)


def test_pending_notebook_is_static_explicit_and_preserves_historical_json():
    original = historical_notebook()
    before = json.dumps(original, sort_keys=True)
    book = report.build_notebook(snapshot(), original)
    assert json.dumps(original, sort_keys=True) == before
    preserved = [cell for cell in book["cells"] if not report._owned(cell)]
    assert json.dumps(preserved, sort_keys=True) == json.dumps(original["cells"], sort_keys=True)
    assert book["metadata"] == original["metadata"]
    assert book["cells"][1]["id"] == "plan15-phase8-progress"
    assert book["cells"][4]["id"] == "plan15-phase8-intro"
    new = [cell for cell in book["cells"] if report._owned(cell)]
    assert all(cell["cell_type"] == "markdown" for cell in new)
    assert sum(bool(cell.get("attachments")) for cell in new) == 1
    assert "Acceptance only" in new[-1]["source"]
    assert "remain pending" in new[-1]["source"]
    assert "Smoke acceptance only" in new[0]["source"]
    nbformat.validate(nbformat.from_dict(deepcopy(book)))


def test_replace_only_owned_cells_and_repeat_publication_is_idempotent():
    previous = historical_notebook()
    previous["cells"].extend([
        dict(cell_type="markdown", metadata={report.TAG: False}, id="stale-phase-cell", source="old phase 8"),
        dict(cell_type="markdown", metadata={"tags": [report.TAG]}, id="stale-tagged", source="old phase 8"),
        dict(cell_type="markdown", metadata={}, id="plan15-phase8-other", source="old phase 8"),
    ])
    first = report.build_notebook(snapshot(), previous)
    second = report.build_notebook(snapshot(), first)
    assert first == second
    assert [cell for cell in second["cells"] if not report._owned(cell)] == historical_notebook()["cells"]


def test_complete_report_has_five_figures_and_no_notebook_analysis(monkeypatch):
    value = snapshot(complete=True)

    def prohibited(*args, **kwargs):
        raise AssertionError("report must not bootstrap")

    monkeypatch.setattr(analysis, "analyze_confirmation", prohibited)
    monkeypatch.setattr(analysis, "paired_interval", prohibited)
    cells = report.build_cells(value)
    assert sum(bool(cell.get("attachments")) for cell in cells) == 5
    assert all(cell["cell_type"] == "markdown" for cell in cells)
    text = "\n".join(cell["source"] for cell in cells)
    assert "Fixed held-out panels" in text
    assert "pointwise" in text and "not simultaneous" in text
    assert "nonsignificance" not in text  # This fixture supports the direct comparison.
    assert "exceeds mean-cost envelope" in text
    assert "16.0 total arrivals" in text and "8.0 total arrivals" in text
    assert "regularizers" in text
    assert "not a causal label-only effect" in text


def test_partial_report_descriptive_without_intervals_and_rejects_false_final():
    value = snapshot(complete=True)
    value["analysis"] = None
    value["confirmation"]["rows"] = [row for row in value["confirmation"]["rows"] if row["replica"] == 0]
    cells = report.build_cells(value)
    text = " ".join(cell["source"] for cell in cells)
    assert "1 complete paired replicas" in text
    assert "descriptive interim means only" in text
    assert "**Pending.**" in text
    value["analysis"] = snapshot(complete=True)["analysis"]
    with pytest.raises(ValueError, match="incomplete"):
        report.build_cells(value)


def test_report_rejects_incompatible_snapshot_and_duplicate_map_rows():
    value = snapshot()
    value["protocol"] = "wrong"
    with pytest.raises(ValueError, match="incompatible"):
        report.build_cells(value)
    value = snapshot(complete=True)
    value["allocation"].append(value["allocation"][0])
    with pytest.raises(ValueError, match="duplicate allocation"):
        report.build_cells(value)


def test_map_and_partial_curves_reject_mismatched_pairing():
    value = snapshot(complete=True)
    value["allocation"][0]["pairing_id"] = "different-pair"
    with pytest.raises(ValueError, match="unpaired allocation"):
        report.build_cells(value)
    value = snapshot(complete=True)
    value["analysis"] = None
    value["allocation"] = []
    value["confirmation"]["rows"][0]["pairing_id"] = "different-pair"
    with pytest.raises(ValueError, match="unpaired partial"):
        report.build_cells(value)


def test_old_notebook_version_updates_header_without_modifying_cells():
    previous = historical_notebook()
    previous["nbformat_minor"] = 2
    book = report.build_notebook(snapshot(), previous)
    assert book["nbformat_minor"] == 5
    assert [cell for cell in book["cells"] if not report._owned(cell)] == previous["cells"]


class FakeStore:
    def __init__(self, root):
        self.root = root
        self.reservations = []

    def check(self):
        return None

    @contextmanager
    def reserve(self, size, **kwargs):
        self.reservations.append(size)
        yield


@pytest.mark.parametrize("internal", [False, True])
def test_atomic_reserved_publish_preserves_source_objects(tmp_path, internal):
    store = FakeStore(tmp_path / "study")
    path = store.root / "reports" / "book.ipynb" if internal else tmp_path / "book.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    original = historical_notebook()
    path.write_text(json.dumps(original))
    assert report.publish(snapshot(), path, store) == path
    actual = json.loads(path.read_text())
    assert [cell for cell in actual["cells"] if not report._owned(cell)] == original["cells"]
    assert store.reservations == [path.stat().st_size]
    assert not list(path.parent.glob(".tmp-*"))


def test_failed_publication_keeps_prior_notebook_and_cleans_staging(tmp_path, monkeypatch):
    path = tmp_path / "book.ipynb"
    original = json.dumps(historical_notebook()).encode()
    path.write_bytes(original)

    def fail_replace(*args):
        raise OSError("simulated interruption")

    monkeypatch.setattr(report.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        report.publish(snapshot(), path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".tmp-*"))


def test_publication_size_bound_and_stop_do_not_modify_history(tmp_path, monkeypatch):
    path = tmp_path / "book.ipynb"
    original = json.dumps(historical_notebook()).encode()
    path.write_bytes(original)
    monkeypatch.setattr(report, "MAX_NOTEBOOK_BYTES", 30)
    with pytest.raises(ValueError, match="bounded"):
        report.publish(snapshot(), path)
    assert path.read_bytes() == original
    with pytest.raises(InterruptedError):
        report.build_cells(snapshot(), check_stop=lambda: True)


def retirement_snapshot():
    value = snapshot(complete=True)
    for row in value["confirmation"]["rows"]:
        for checkpoint in row["metrics"]:
            checkpoint["observations"] *= 10
        row["observations"] = row["metrics"][-1]["observations"]
        row["retired_audit"] = []
        for checkpoint in row["metrics"][1:]:
            for event in range(1, min(checkpoint["n"], 8) + 1):
                exposure = 40 * event - 1
                retired = exposure + 1 + row["capacity"]
                age = checkpoint["observations"] - retired
                if age >= 0:
                    row["retired_audit"].append(dict(
                        n=checkpoint["n"], exposure=exposure, observations=checkpoint["observations"],
                        retirement_observations=retired, age_since_retirement=age,
                        squared_error=float(age + row["replica"] + 1)))
    value["analysis"] = analysis.analyze_confirmation(value["confirmation"]["rows"], value["confirmation"], 17, 40)
    return value


def test_retirement_audit_uses_exact_actual_ages_and_counts_without_imputation():
    value = retirement_snapshot()
    curves = report._retirement_summary(value)
    fd = curves[analysis.FD]
    assert fd["age"] == [24, 64, 104]
    assert fd["counts"] == [12, 8, 4]
    assert fd["mean_mse"] == pytest.approx([26.5, 66.5, 106.5])
    assert fd["unique_observations"] == 12
    boundary = curves["ewc_only"]
    assert boundary["age"] == [0, 40, 80, 120]
    assert boundary["counts"] == [16, 12, 8, 4]
    assert boundary["unique_observations"] == 16


def test_retirement_panel_stays_inside_existing_five_figure_report():
    cells = report.build_cells(retirement_snapshot())
    assert sum(bool(cell.get("attachments")) for cell in cells) == 5
    effects = next(cell for cell in cells if cell["id"] == "plan15-phase8-effects")
    assert "first eight novel arrivals" in effects["source"]
    assert "actual retirement" in effects["source"]
    assert "not independent sample sizes" in effects["source"]
    assert "Different capacities" in effects["source"]
    assert "not a paired causal contrast" in effects["source"]


@pytest.mark.parametrize("defect", ["duplicate", "wrong_age", "wrong_checkpoint", "negative_error", "over_eight"])
def test_retirement_audit_rejects_malformed_counts_and_clocks(defect):
    value = retirement_snapshot()
    row = value["confirmation"]["rows"][0]
    if defect == "duplicate":
        row["retired_audit"].append(deepcopy(row["retired_audit"][0]))
    elif defect == "wrong_age":
        row["retired_audit"][0]["age_since_retirement"] += 1
    elif defect == "wrong_checkpoint":
        row["retired_audit"][0]["n"] = 99
    elif defect == "negative_error":
        row["retired_audit"][0]["squared_error"] = -1.
    else:
        row["retired_audit"] = [dict(n=4, exposure=i, observations=160, retirement_observations=i + 17,
                                     age_since_retirement=143 - i, squared_error=1.) for i in range(9)]
    with pytest.raises(ValueError):
        report._retirement_summary(value)


def test_partial_retirement_audit_is_visible_without_bootstrap_or_final_claims():
    value = retirement_snapshot()
    value["analysis"] = None
    value["confirmation"]["rows"] = value["confirmation"]["rows"][:1]
    cells = report.build_cells(value)
    effects = next(cell for cell in cells if cell["id"] == "plan15-phase8-effects")
    assert "No scientific inference" in effects["source"]
    assert "nominal paired 95%" not in effects["source"]
    assert "**Pending.**" in cells[-1]["source"]


def test_progress_counts_external_bytes_in_shared_use_without_counting_reservations_twice():
    value = snapshot()
    value["progress"]["storage"] = dict(used_bytes=2_000_000_000, external_bytes=23_000_000_000,
                                        reserved_bytes=1_000_000_000, max_bytes=500_000_000_000)
    text = report._progress(value)["source"]
    assert "25 GB used / 500 GB cap" in text
    assert "this study 2 GB + external 23 GB" in text
    assert "1 GB reserved" in text
    assert "26 GB used" not in text
    value["progress"]["storage"].pop("external_bytes")
    assert "Shared storage: pending" in report._progress(value)["source"]


def test_significant_changes_are_bounded_prose_not_raw_dictionaries():
    value = snapshot()
    value["changes"] = [dict(stage="8B", change="Shared solver repair", reason="Objective gap exceeded tolerance",
                             before=dict(max_iter=100, max_eval=150), after=dict(max_iter=200, max_eval=300),
                             validity="Development only", validation=dict(passed=12, failed=0))] * 4
    cell = next(cell for cell in report.build_cells(value) if cell["id"] == "plan15-phase8-changes")
    text = cell["source"]
    assert text.count("Shared solver repair") == 3
    assert "max iter 100 -> 200" in text and "max eval 150 -> 300" in text
    assert "passed: 12; failed: 0" in text
    assert "Development only" in text
    assert "{'" not in text and "'reason':" not in text


def test_acceptance_scaffold_is_optional_concise_and_never_scientific_smoke_inference():
    value = retirement_snapshot()
    value["smoke"] = True
    value["acceptance"] = dict(status="passed", summary="CPU recovery checks completed.",
                               tests=[dict(name=f"Acceptance check {i}", result=dict(passed=1, failed=0), seconds=.1)
                                      for i in range(8)])
    original = historical_notebook()
    book = report.build_notebook(value, original)
    assert [cell for cell in book["cells"] if not report._owned(cell)] == original["cells"]
    acceptance = next(cell for cell in book["cells"] if cell["id"] == "plan15-phase8-acceptance")
    assert "Acceptance check 5" in acceptance["source"]
    assert "Acceptance check 6" not in acceptance["source"]
    assert "Additional checks: 2" in acceptance["source"]
    assert "passed: 1; failed: 0" in acceptance["source"]
    text = " ".join(cell["source"] for cell in book["cells"] if report._owned(cell))
    assert "supports FD advantage" not in text
    assert "nominal 95%" not in text
    assert "**Acceptance only.**" in book["cells"][-1]["source"]
    assert sum(bool(cell.get("attachments")) for cell in book["cells"] if report._owned(cell)) <= 5


def test_acceptance_supports_a_compact_tests_result_mapping():
    value = snapshot()
    value["acceptance"] = dict(tests={"Unit suite": "168 passed", "Interrupted CPU smoke": "recovered"})
    cell = report._acceptance(value)
    assert "| Unit suite | 168 passed |" in cell["source"]
    assert "| Interrupted CPU smoke | recovered |" in cell["source"]


def test_sealed_acceptance_recorder_schema_shows_tests_both_smokes_and_charge():
    value = snapshot()
    value["acceptance"] = dict(status="8A complete", total_tests=168, production_started=False,
        charged_evidence_seconds=150.,
        tests=[dict(path="cache/acceptance/phase8-unit.xml", tests=168, failures=0, errors=0, skipped=0,
                    process_seconds=12., sha256="not displayed")],
        smokes=[dict(device=device, units=30, novel_count=3, runtime_seconds=60., seals={"hidden": "digest"})
                for device in ("cpu", "cuda")])
    text = report._acceptance(value)["source"]
    assert "168 tests; 0 failures, 0 errors, 0 skipped" in text
    assert "CPU recovery smoke" in text and "CUDA recovery smoke" in text
    assert "30 units, 3 novel observations" in text
    assert "Charged acceptance compute: 150 seconds" in text
    assert "Production has not started" in text
    assert "pending" not in text
    assert "sha256" not in text and "seals" not in text
