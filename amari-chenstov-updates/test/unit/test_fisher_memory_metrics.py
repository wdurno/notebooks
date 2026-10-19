"""Pure statistical contracts and artifact-only Plan 15 reporting tests."""

import contextlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pytest

from mnist_experiment.fisher_memory import metrics, report


def _logits(labels, predicted=None):
    predicted = labels if predicted is None else predicted
    logits = np.full((len(labels), 10), -2.0)
    logits[np.arange(len(labels)), predicted] = 3.0
    return logits


def test_predictive_metrics_use_conditional_rates_and_fixed_prevalence():
    labels = [9, 9, 0, 1]
    logits = _logits(labels, [9, 0, 9, 1])
    result = metrics.predictive_metrics(logits, labels)
    assert result["accuracy"] == .5
    assert result["nine_recall"] == .5
    assert result["nine_fpr"] == .5
    assert result["nine_precision"] == pytest.approx(.1)
    assert result["nine_precision_empirical"] == .5
    assert result["false_positives_per_1000"] == 500
    assert result["nine_count"] == 2
    assert not result["no_positive_labels"]
    assert np.isfinite(result["nll"])
    json.dumps(result, allow_nan=False)


def test_weighted_metrics_and_stratum_standardization():
    labels = [9, 9, 0, 1]
    logits = _logits(labels, [9, 0, 9, 1])
    result = metrics.predictive_metrics(logits, labels, [3, 1, 2, 2])
    assert result["nine_recall"] == .75
    assert result["nine_fpr"] == .5
    assert result["accuracy"] == 5 / 8
    assert result["nine_precision"] == pytest.approx(.075 / (.075 + .45))
    # Changing total stratum mass does not change class-conditional rates.
    shifted = metrics.predictive_metrics(logits, labels, [30, 10, 2, 2])
    assert shifted["nine_precision"] == result["nine_precision"]
    assert shifted["nll"] != result["nll"]


def test_missing_positive_recall_is_not_zero_and_precision_flag_is_explicit():
    result = metrics.predictive_metrics(_logits([0, 1]), [0, 1])
    assert result["nine_recall"] is None
    assert result["nine_precision"] == 0
    assert result["no_predicted_positives"]
    assert result["no_positive_labels"]
    assert result["recall_missing_reason"]
    result = metrics.predictive_metrics(_logits([0], [9]), [0])
    assert result["nine_precision"] is None
    assert not result["no_predicted_positives"]


def test_zero_evaluation_weight_is_not_positive_support():
    result = metrics.predictive_metrics(_logits([9, 0]), [9, 0], [0, 1])
    assert not result["no_positive_labels"]
    assert result["no_positive_support"]
    assert result["nine_recall"] is None


def test_separate_conditional_weights_preserve_prevalence_zero_evaluation():
    logits = _logits([9, 0], [9, 9])
    ordinary = metrics.predictive_metrics(logits, [9, 0], [0, 1])
    fixed = metrics.predictive_metrics(logits, [9, 0], [0, 1], conditional_weights=[.1, .9])
    assert fixed["nll"] == ordinary["nll"]
    assert fixed["accuracy"] == ordinary["accuracy"]
    assert ordinary["nine_recall"] is None
    assert fixed["nine_recall"] == 1
    assert fixed["nine_fpr"] == 1
    assert fixed["nine_precision"] == pytest.approx(.1)
    assert fixed["nine_nll"] is not None


def test_empty_predictions_are_explicit_and_json_friendly():
    result = metrics.predictive_metrics(np.empty((0, 10)), [])
    assert result["empty"]
    assert result["nll"] is None
    assert result["accuracy"] is None
    assert result["nine_recall"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [1], [np.nan, 1]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        metrics.predictive_metrics(_logits([0, 9]), [0, 9], weights)


@pytest.mark.parametrize("labels", [[1.5], [-1], [10], [np.nan]])
def test_invalid_labels_rejected(labels):
    with pytest.raises(ValueError, match="labels"):
        metrics.predictive_metrics(np.zeros((1, 10)), labels)


@pytest.mark.parametrize("prevalence", [0, 1, -.1, np.nan, True])
def test_invalid_prevalence_rejected(prevalence):
    with pytest.raises(ValueError, match="p_ref"):
        metrics.predictive_metrics(_logits([9]), [9], p_ref=prevalence)


def test_metrics_accept_torch_without_retaining_gradients():
    import torch

    logits = torch.zeros(2, 10, requires_grad=True)
    result = metrics.predictive_metrics(logits, torch.tensor([0, 9]))
    assert result["nll"] == pytest.approx(np.log(10))
    assert logits.grad is None


def test_pr_curve_handles_ties_and_prevalence_standardized_ap():
    result = metrics.standardized_pr_curve([.9, .8, .7, .6], [9, 0, 9, 0])
    assert result["recall"] == [0, .5, .5, 1, 1]
    assert result["fpr"] == [0, 0, .5, .5, 1]
    assert result["precision"][0] == 0
    assert result["average_precision"] == pytest.approx(.5 + .5 * .1 / .55)
    tied = metrics.standardized_pr_curve([.5, .5, .5, .5], [9, 0, 9, 0])
    assert tied["thresholds"] == [None, .5]
    assert tied["average_precision"] == pytest.approx(.1)
    perfect = metrics.standardized_pr_curve([.9, .8, .2, .1], [9, 9, 0, 0])
    assert perfect["average_precision"] == 1
    json.dumps(result, allow_nan=False)


def test_pr_strata_missingness_and_weighted_equivalence():
    result = metrics.standardized_pr_curve([.4, .1], [0, 1])
    assert result["average_precision"] is None
    assert result["missing_reason"]
    a = metrics.standardized_pr_curve([.9, .8, .2, .1], [9, 0, 9, 0])
    b = metrics.standardized_pr_curve([.9, .8, .2, .1], [9, 0, 9, 0], [100, 1, 100, 1])
    assert a == b


def test_pr_accepts_probability_matrix_not_logits():
    p = np.full((2, 10), .01)
    p[:, 9] = .91
    curve = metrics.standardized_pr_curve(p, [9, 0])
    assert curve["average_precision"] == pytest.approx(.1)
    with pytest.raises(ValueError, match="sum to one"):
        metrics.standardized_pr_curve(np.ones((2, 10)), [9, 0])


def test_pr_cumulative_roundoff_never_exceeds_probability_bounds():
    rng = np.random.default_rng(2)
    for _ in range(20):
        labels = np.r_[np.full(32, 9), np.zeros(80)]
        weights = rng.uniform(.00001, 1, len(labels))
        curve = metrics.standardized_pr_curve(rng.uniform(size=len(labels)), labels, weights)
        for field in ("recall", "fpr", "precision"):
            assert np.all(np.asarray(curve[field]) <= 1)
            assert np.all(np.asarray(curve[field]) >= 0)
        assert 0 <= curve["average_precision"] <= 1


def test_large_finite_logits_do_not_underflow_nll():
    logits = np.full((1, 10), -1000.)
    logits[0, 0] = 1000
    result = metrics.predictive_metrics(logits, [9])
    assert result["nll"] == 2000
    assert np.isfinite(result["nll"])


def test_normalized_auc_respects_irregular_exposure_grid():
    assert metrics.normalized_auc([0, 2, 10], [0, 2, 10]) == 5
    assert metrics.normalized_auc([5], [3]) is None
    assert metrics.normalized_auc([], []) is None
    with pytest.raises(ValueError, match="strictly increasing"):
        metrics.normalized_auc([0, 0, 1], [1, 2, 3])
    with pytest.raises(ValueError, match="finite"):
        metrics.normalized_auc([0, 1], [1, np.nan])


def test_retention_auc_right_censors_without_extrapolation():
    result = metrics.cohort_retention_auc([0, 10, 20, 30], [0, 10, 20, 30],
                                         eviction_exposure=10, end_exposure=30, capacity=16)
    assert result["auc"] == 20
    assert result["declared_horizon"] == 20
    assert result["observed_horizon"] == 20
    assert result["right_censored"]
    assert result["complete_for_declared_horizon"]
    partial = metrics.cohort_retention_auc([0, 10, 20], [0, 10, 20],
                                          eviction_exposure=10, end_exposure=40, capacity=16)
    assert partial["auc"] == 15
    assert partial["observed_horizon"] == 10
    assert not partial["complete_for_declared_horizon"]


def test_retention_zero_horizon_and_unobserved_left_endpoint():
    result = metrics.cohort_retention_auc([0, 10], [1, 2], eviction_exposure=10,
                                         end_exposure=10, capacity=16)
    assert result["auc"] is None
    assert result["missing_reason"] == "no post-eviction follow-up"
    result = metrics.cohort_retention_auc([12, 20], [1, 2], eviction_exposure=10,
                                         end_exposure=30, capacity=16)
    assert result["auc"] is None
    assert result["right_censored"]


def test_cohort_summary_equal_weights_and_partial_exclusion():
    result = metrics.cohort_summary([
        {"auc": 1., "complete_for_declared_horizon": True, "right_censored": False},
        {"auc": 3., "complete_for_declared_horizon": True, "right_censored": True},
        {"auc": 100., "complete_for_declared_horizon": False, "right_censored": True},
    ])
    assert result["mean_auc"] == 2
    assert result["usable_cohorts"] == 2
    assert result["right_censored_cohorts"] == 2


def test_paired_summary_resamples_replicas_deterministically():
    a = {0: 1., 1: 2., 2: 4., 3: None, 4: 5.}
    b = {0: .5, 1: 2.5, 2: 3., 3: 1., 5: 0.}
    summary = metrics.paired_summary(a, b, seed=123, resamples=333)
    assert summary == metrics.paired_summary(a, b, seed=123, resamples=333)
    assert summary["n_pairs"] == 3
    assert summary["mean_gain"] == pytest.approx(1 / 3)
    assert summary["median_gain"] == .5
    assert summary["favorable_replicas"] == 2
    assert summary["unmatched_control"] == summary["unmatched_treatment"] == 1
    assert summary["missing_pairs"] == 1
    assert summary["small_sample"]
    assert len(summary["paired_values"]) == 3
    assert metrics.paired_summary({0: 1.}, {0: .5})["ci"] is None
    assert metrics.paired_summary({}, {})["mean_gain"] is None
    assert metrics.deterministic_seed(1, "bootstrap") != metrics.deterministic_seed(1, "probes")


def test_functional_dependence_uses_signed_loss_and_directed_kl():
    base = _logits([0, 9])
    changed = _logits([0, 9], [1, 9])
    result = metrics.dependence_metrics(base, changed, [0, 9])
    assert result["nll_difference"] == pytest.approx(2.5)
    assert result["predictive_kl"] > 0
    reverse = metrics.dependence_metrics(changed, base, [0, 9])
    assert reverse["nll_difference"] == pytest.approx(-2.5)
    same = metrics.dependence_metrics(base, base, [0, 9])
    assert same["predictive_kl"] == same["nll_difference"] == 0
    vulnerable = metrics.future_vulnerability(base, changed, [0, 9])
    assert vulnerable["vulnerability_nll"] == result["nll_difference"]


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def _metadata(**overrides):
    return {"schema_version": 1, "kind": "trajectory", "phase": "phase3", "environment": "rotation_linear",
            "condition": "replay_ewc", "replica": 0, "config": {"strength": 0},
            "replay_capacity": 4, "end_exposure": 16, **overrides}


def _rows(offset=0.):
    rows = []
    for exposure in (0, 4, 8, 12, 16):
        rows.append({"schema_version": 1, "step": exposure // 4, "exposure": exposure,
                     "split": "current", "nll": 1. - exposure / 100 + offset, "accuracy": .5 + exposure / 100,
                     "nine_recall": .4, "nine_fpr": .1, "nine_precision": .3})
        if exposure >= 8:
            for split in ("historical", "evicted_train"):
                rows.append({"schema_version": 1, "step": exposure // 4, "exposure": exposure,
                             "split": split, "cohort_id": 0, "eviction_exposure": 8,
                             "cohort_age": exposure - 8,
                             "nll": 1. + offset, "accuracy": .6,
                             "nine_recall": .3, "nine_precision": .25})
    return rows


def _write_unit(root, name, *, metadata=None, rows=None, complete=True):
    path = root / "units" / name
    meta = _metadata() if metadata is None else metadata
    _write_json(path / "metadata.json", meta)
    _write_json(path / "metrics.json", _rows() if rows is None else rows)
    if complete:
        (path / "COMPLETED").write_text("complete\n")
    return path


def test_empty_report_is_explicit_and_creates_no_files(tmp_path):
    notebook = report.build_notebook(tmp_path)
    nbformat.validate(notebook)
    assert "Execution progress" in notebook.cells[0].source
    assert "No sealed results yet" in notebook.cells[0].source
    assert notebook.metadata.plan15.artifact_only
    assert not list(tmp_path.iterdir())
    assert "New/current holdout" in " ".join(c.source for c in notebook.cells)


def test_report_ignores_unsealed_partial_metrics(tmp_path):
    path = _write_unit(tmp_path, "in-progress", complete=False)
    (path / "metrics.json").write_text("not valid JSON yet")
    data = report.collect_artifacts(tmp_path)
    assert data["units"] == []
    assert data["pending"] == 1
    _write_json(path / "progress.json", {"schema_version": 1, "sealed": True, "rows": _rows()})
    data = report.collect_artifacts(tmp_path)
    assert len(data["units"]) == 1
    assert not data["units"][0]["complete"]


def test_nontrajectory_manifests_are_skipped_before_metrics_requirements(tmp_path):
    for kind in ("assets", "decision", "calibration"):
        path = tmp_path / "units" / kind
        _write_json(path / "manifest.json", {"metadata": {"kind": kind}})
        (path / "COMPLETED").write_text("complete")
    data = report.collect_artifacts(tmp_path)
    assert data["units"] == []
    assert data["pending"] == 0


def test_trajectory_manifest_metadata_is_authoritative(tmp_path):
    path = _write_unit(tmp_path, "unit")
    _write_json(path / "manifest.json", {"metadata": _metadata(environment="rotation_sigmoid")})
    data = report.collect_artifacts(tmp_path)
    assert data["units"][0]["metadata"]["environment"] == "rotation_sigmoid"


@pytest.mark.parametrize("damage", ["missing", "corrupt", "schema", "duplicate", "nonfinite", "mismatch"])
def test_corrupt_completed_units_fail_clearly(tmp_path, damage):
    path = _write_unit(tmp_path, "sealed")
    if damage == "missing":
        (path / "metrics.json").unlink()
    elif damage == "corrupt":
        (path / "metrics.json").write_text("{")
    elif damage == "nonfinite":
        (path / "metrics.json").write_text('[{"nll":NaN}]')
    else:
        rows = _rows()
        if damage == "schema":
            rows[0]["schema_version"] = 2
        elif damage == "mismatch":
            rows[0]["environment"] = "digit9_mixture"
        else:
            rows.append(dict(rows[0]))
        _write_json(path / "metrics.json", rows)
    with pytest.raises(report.ArtifactError):
        report.build_notebook(tmp_path)


def test_explicit_row_adapter_does_not_silently_infer_schema(tmp_path):
    rows = _rows()
    rows[0]["exposures"] = rows[0].pop("exposure")
    _write_unit(tmp_path, "legacy", rows=rows)
    with pytest.raises(report.ArtifactError, match="exposure"):
        report.collect_artifacts(tmp_path)

    def adapter(row, metadata):
        if "exposures" in row:
            row["exposure"] = row.pop("exposures")
        return row

    assert report.collect_artifacts(tmp_path, row_adapter=adapter)["units"]


def test_paired_retention_requires_matching_observed_horizons(tmp_path):
    _write_unit(tmp_path, "control", metadata=_metadata())
    _write_unit(tmp_path, "treatment", metadata=_metadata(condition="jvp"), rows=_rows(-.1))
    data = report.collect_artifacts(tmp_path)
    conclusion = report._conclusion(data["units"], {"confirmation_complete": True}).source
    assert "Historical holdout | 1 | 0.1" in conclusion
    assert "New/current holdout | 1 | 0.1" in conclusion
    rows = _rows(-.1)
    rows = [r for r in rows if r["split"] != "historical" or r["exposure"] != 16]
    _write_json(tmp_path / "units" / "treatment" / "metrics.json", rows)
    conclusion = report._conclusion(report.collect_artifacts(tmp_path)["units"], {}).source
    assert "Historical holdout | 0 | pending" in conclusion


def test_confirmation_readout_distinguishes_frozen_reference_and_actual_compute(tmp_path):
    for replica in (0, 1):
        for condition, offset, seconds in (("replay_ewc", 0., 100.),
                                           ("finite_difference", -.1, 200.),
                                           ("replay_compute_matched", 0., 120.)):
            rows = _rows(offset)
            for row in rows:
                row["cumulative_active_seconds"] = seconds
                if row["split"] == "current":
                    row["frozen_nll"] = .8
            _write_unit(tmp_path, f"{condition}-{replica}", rows=rows,
                        metadata=_metadata(condition=condition, replica=replica))
    units = report.collect_artifacts(tmp_path)["units"]
    source = report._conclusion(units, {"confirmation_complete": True,
                                       "conclusion": "persistent under-solving warnings"}).source
    assert "nominal 95% paired intervals favor the treatment on both NLL endpoints" in source
    assert "0.82 versus 0.8 for the frozen initial model (descriptive)" in source
    assert "online time 2.00x replay/EWC" in source
    assert "60%--60%" in source
    assert "not a realized equal-time comparison" in source
    assert "Mechanism remains unresolved" in source
    assert "not exact objective minimization" in source
    partial = report._conclusion(units, {}).source
    assert "nominal 95% paired intervals favor" not in partial
    assert "pending fixed-cohort completion" in partial


def test_confirmation_readout_does_not_label_a_null_candidate_favorable(tmp_path):
    for replica in (0, 1):
        for condition in ("replay_ewc", "finite_difference"):
            _write_unit(tmp_path, f"{condition}-{replica}", metadata=_metadata(
                condition=condition, replica=replica))
    source = report._conclusion(report.collect_artifacts(tmp_path)["units"],
                                {"confirmation_complete": True}).source
    assert "do not establish improvement on both NLL endpoints" in source
    assert "nominal 95% paired intervals favor" not in source


def test_report_rendering_is_bounded_separate_and_artifact_only(tmp_path, monkeypatch):
    for env in report.ENVIRONMENTS:
        for replica in (0, 1):
            for condition in ("replay_ewc", "jvp"):
                name = f"{env}-{condition}-{replica}"
                meta = _metadata(environment=env, condition=condition, replica=replica, expected_nines=10.1)
                path = _write_unit(tmp_path, name, metadata=meta, rows=_rows(-.1 if condition == "jvp" else 0))
                (path / "checkpoint.pt").write_bytes(b"do not read me")
    _write_json(tmp_path / "status.json", {"phase": "phase3", "active_seconds": 3600,
                                          "used_bytes": 1000, "max_bytes": 500_000_000_000,
                                          "changes": [{"revision": "v2", "problem": "fixed sign", "validation": "passed"}]})
    original = Path.open

    def forbid_checkpoints(path, *args, **kwargs):
        assert path.suffix != ".pt", "report tried to load a checkpoint"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_checkpoints)
    notebook = report.build_notebook(tmp_path)
    nbformat.validate(notebook)
    figures = [out for cell in notebook.cells for out in cell.get("outputs", []) if "image/png" in out.get("data", {})]
    assert 3 <= len(figures) <= 8
    sources = "\n".join(c.source for c in notebook.cells)
    assert sources.index("Calibration and pilot") < sources.index("Rotation results") < sources.index("Digit-9 mixture results")
    assert "Expected online 9s per trajectory: **10.1**" in sources
    assert "Significant changes" in sources
    assert "pending fixed-cohort completion" in sources
    assert str(tmp_path) not in nbformat.writes(notebook)
    assert len(nbformat.writes(notebook).encode()) < report.MAX_NOTEBOOK_BYTES


def test_duplicate_strengths_are_not_silently_pooled(tmp_path):
    _write_unit(tmp_path, "one")
    _write_unit(tmp_path, "two")
    with pytest.raises(report.ArtifactError, match="duplicate phase"):
        report.build_notebook(tmp_path)


def test_report_accepts_only_negligible_probability_roundoff(tmp_path):
    rows = _rows()
    rows[0]["nine_fpr"] = 1.0000000000000002
    _write_unit(tmp_path, "roundoff", rows=rows)
    assert report.collect_artifacts(tmp_path)["units"][0]["rows"][0]["nine_fpr"] == 1
    rows[0]["nine_fpr"] = 1.001
    _write_json(tmp_path / "units" / "roundoff" / "metrics.json", rows)
    with pytest.raises(report.ArtifactError, match="nine_fpr"):
        report.collect_artifacts(tmp_path)


def test_smoke_disclaimer_and_execution_failure_are_prominent(tmp_path):
    _write_unit(tmp_path, "smoke", metadata=_metadata(smoke=True))
    _write_json(tmp_path / "status.json", {"status": "failed", "failure": "test failure", "note": "Smoke acceptance only"})
    notebook = report.build_notebook(tmp_path)
    assert "SMOKE ACCEPTANCE ONLY" in notebook.cells[0].source
    assert "test failure" in notebook.cells[0].source
    assert "Acceptance evidence only; no scientific result" in notebook.cells[-1].source


def test_pre_eviction_rows_remain_separate_from_retention(tmp_path):
    rows = _rows()
    rows.extend([{"schema_version": 1, "step": 0, "exposure": 0, "split": split,
                  "cohort_id": 0, "nll": 100., "accuracy": 0.} for split in
                 ("cohort_pre_eviction", "historical_pre_eviction")])
    _write_unit(tmp_path, "cohort", rows=rows)
    unit = report.collect_artifacts(tmp_path)["units"][0]
    auc, _ = report._auc_for_unit(unit, "historical")
    assert auc == 1.


def test_refresh_calls_publisher_once_and_requires_reservation(tmp_path):
    calls = []
    target = tmp_path / "reports" / "test.ipynb"
    actual = report.refresh(tmp_path, target, publish=lambda path, content: calls.append((path, content)))
    assert actual == target
    assert len(calls) == 1
    nbformat.validate(nbformat.reads(calls[0][1].decode(), as_version=4))
    assert not target.exists()
    with pytest.raises(RuntimeError, match="storage-reserved"):
        report.refresh(tmp_path)


class _DummyStore:
    def __init__(self, root):
        self.root = root
        self.reservations = []

    @contextlib.contextmanager
    def reserve(self, amount, *, finalizing=False):
        self.reservations.append(amount)
        yield


def test_reserved_publication_is_atomic_and_bounded(tmp_path, monkeypatch):
    store = _DummyStore(tmp_path)
    publish = report.reserved_publisher(store)
    target = tmp_path / "reports" / "test.ipynb"
    publish(target, b"old")
    assert target.read_bytes() == b"old"
    assert store.reservations == [3]

    def failed_replace(*args):
        raise OSError("simulated interrupted publish")

    monkeypatch.setattr(report.os, "replace", failed_replace)
    with pytest.raises(OSError, match="interrupted"):
        publish(target, b"new")
    assert target.read_bytes() == b"old"
    assert list(target.parent.iterdir()) == [target]
    with pytest.raises(ValueError, match="reports"):
        publish(tmp_path / "units" / "bad.ipynb", b"bad")


def test_predictive_plot_shows_one_paired_frozen_reference_from_stored_metrics(monkeypatch):
    from matplotlib import pyplot as plt
    units = []
    for replica in (0, 1):
        for condition in ("replay_ewc", "jvp"):
            rows = _rows()
            for row in rows:
                if row["split"] == "current":
                    row["frozen_nll"] = .4 + .2 * replica
            units.append({"metadata": _metadata(replica=replica, condition=condition), "rows": rows})
    monkeypatch.setattr(report, "_plot_cell", lambda figure, caption: figure)
    figure = report._predictive_figure(units, "rotation_linear", "phase3")
    try:
        frozen = [line for line in figure.axes[0].lines if line.get_label() == "Frozen initial model"]
        assert len(frozen) == 1
        np.testing.assert_allclose(frozen[0].get_ydata(), .5)
        assert frozen[0].get_linestyle() == "--"
        assert not any(line.get_label() == "Frozen initial model" for axis in figure.axes[1:] for line in axis.lines)
    finally:
        plt.close(figure)


def test_actual_store_report_publication_and_cli(tmp_path):
    from mnist_experiment.fisher_memory.artifacts import Store

    root = tmp_path / "study"
    with Store(root, {"source_hash": "test"}, free_reserve_bytes=0,
               finalization_reserve_bytes=32768) as store:
        store.begin("trajectory", _metadata())
        store.write_json("trajectory", "metrics.json", _rows())
        store.complete("trajectory")
        path = report.refresh(root, publish=report.reserved_publisher(store))
        nbformat.validate(nbformat.read(path, as_version=4))
        assert store.storage_status()["used_bytes"] >= path.stat().st_size
        assert store.storage_status()["reserved_bytes"] == 0
    report.main(["--root", str(root), "--budget-root", str(tmp_path),
                 "--data-root", str(tmp_path / "absent-dataset")])
    assert path.exists()
    metric_path = root / "units" / "trajectory" / "metrics.json"
    metric_path.write_bytes(metric_path.read_bytes().replace(b'"accuracy":0.5', b'"accuracy":0.6', 1))
    with pytest.raises(report.ArtifactError, match="checksum"):
        report.build_notebook(root)


def test_report_cli_locks_shared_budget_and_recounts_sibling_storage(tmp_path, monkeypatch):
    from mnist_experiment.fisher_memory.artifacts import Store
    from mnist_experiment.fisher_memory.run import budget_lock, file_bytes

    budget, root = tmp_path / "budget", tmp_path / "budget/study"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "images").write_bytes(b"x" * 1024)
    with Store(root, {"source_hash": "test"}, free_reserve_bytes=0) as store:
        store.write_json(None, "provenance.json", {"dataset_manifest": {"images": {"bytes": 2048}}})
    argv = ["--root", str(root), "--budget-root", str(budget), "--data-root", str(dataset)]
    before = (root / "byte_ledger.json").read_bytes()
    with budget_lock(budget), pytest.raises(RuntimeError, match="shared storage budget"):
        report.main(argv)
    assert (root / "byte_ledger.json").read_bytes() == before
    sibling = budget / "another-study"
    sibling.mkdir()
    (sibling / "checkpoint").write_bytes(b"x" * 4096)
    monkeypatch.setattr(report, "refresh", lambda *args, **kwargs: None)
    report.main(argv)
    ledger = json.loads((root / "byte_ledger.json").read_text())
    assert ledger["external_bytes"] >= 2048 + file_bytes(sibling) + report.MAX_NOTEBOOK_BYTES
    with pytest.raises(SystemExit):
        report.main(["--root", str(root), "--budget-root", str(sibling)])


def test_flattened_threshold_rows_and_compute_summaries(tmp_path):
    rows = _rows()
    curve = metrics.standardized_pr_curve([.9, .8, .2, .1], [9, 0, 9, 0])
    for population in ("current", "historical"):
        for threshold, recall, precision in zip(curve["thresholds"], curve["recall"], curve["precision"]):
            rows.append({"schema_version": 1, "step": 4, "exposure": 16, "split": "pr", "population": population,
                         "cohort_id": 0, "eviction_exposure": 8, "threshold": threshold,
                         "nine_recall": recall, "nine_precision": precision,
                         "average_precision": curve["average_precision"], "p_ref": .1})
    for row in rows:
        row["cumulative_active_seconds"] = row["exposure"] * 2
    _write_unit(tmp_path, "pr", metadata=_metadata(environment="digit9_mixture", expected_nines=10.1), rows=rows)
    unit = report.collect_artifacts(tmp_path)["units"][0]
    assert report._pr_records(unit, "historical")[0]["pr_curve"]["recall"] == curve["recall"]
    notebook = report.build_notebook(tmp_path)
    assert "Sample and compute efficiency" in "\n".join(c.source for c in notebook.cells)
    assert sum(len(c.get("outputs", [])) for c in notebook.cells) <= 7


def test_screen_display_uses_middle_strength_not_best_outcome_and_human_labels(tmp_path):
    for strength in (.01, .1, 1., 10.):
        for replica in (0, 1):
            metadata = _metadata(phase="phase2", family="pilot_screen", condition="jvp", replica=replica,
                                 treatment_id=f"jvp-hash-{strength}", config={"strength": strength})
            _write_unit(tmp_path, f"s-{strength}-{replica}", metadata=metadata, rows=_rows(-.8 if strength == 10 else 0))
    units = report.collect_artifacts(tmp_path)["units"]
    selected, notes = report._selected_groups(units)
    assert len(selected) == 2
    assert {u["metadata"]["config"]["strength"] for u in selected} == {.1}
    assert "NOT the best outcome" in " ".join(notes)
    assert set(report._display_labels(selected).values()) == {"JVP (lambda=0.1)"}
    for unit in units:
        unit["metadata"]["family"] = "pilot_validation"
    assert len(report._selected_groups(units)[0]) == 8


def test_screen_display_prefers_published_common_baseline(tmp_path):
    for ridge in (.01, .1):
        _write_unit(tmp_path, f"baseline-{ridge}", metadata=_metadata(
            phase="phase1", family="policy_development", treatment_id=f"baseline-{ridge}",
            config={"strength": 0, "ridge_multiplier": ridge}))
    selected, _ = report._selected_groups(report.collect_artifacts(tmp_path)["units"], {
        "common_policy": {"rotation_linear": {"strength": 0, "ridge_multiplier": .1}}})
    assert len(selected) == 1
    assert selected[0]["metadata"]["config"]["ridge_multiplier"] == .1


def test_predictive_axes_separate_exposure_from_post_eviction_age(tmp_path, monkeypatch):
    from matplotlib import pyplot as plt

    _write_unit(tmp_path, "axes")
    unit = report.collect_artifacts(tmp_path)["units"][0]
    labels = []

    def inspect(figure, caption):
        labels.extend(axis.get_xlabel() for axis in figure.axes)
        assert "Right censoring" in caption
        plt.close(figure)
        return None

    monkeypatch.setattr(report, "_plot_cell", inspect)
    report._predictive_figure([unit], "rotation_linear", "phase3")
    assert labels[3:] == ["Unique digits integrated", "Digits since cohort eviction", "Digits since cohort eviction"]
