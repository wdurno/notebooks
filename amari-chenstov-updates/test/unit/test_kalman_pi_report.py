"""Scalar-only Plan 17 analysis and artifact-safe CPU publication tests."""

import base64
from contextlib import contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys

import nbformat
import numpy as np
from PIL import Image
import pytest

from mnist_experiment.kalman_pi import analysis, report

ARMS = ("fixed", "covariance", "ema_h16", "kalman_eta0.001")


@pytest.fixture(autouse=True)
def plotting_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))


def trajectory(condition="fixed", replica=0, *, stage="fresh", schedule="main", c=.025,
               ridge=.1, completed=True, delta=.02):
    correction = 0 if condition == "fixed" else delta
    rows = []
    for i, arrivals in enumerate((0, 4, 12, 20)):
        row = dict(step=arrivals // 4, arrivals=arrivals, p=.005 if i < 2 else .1,
            segment="low_hold" if i < 2 else "rise", expected_nines=.1 * arrivals, observed_nines=arrivals // 4,
            nine_recall=.2 + .1 * i + correction * i, old_accuracy=.9 - .02 * i - correction * i / 4,
            frozen_nine_recall=.2, frozen_old_accuracy=.9, nine_fpr=.01, nine_precision=None if i == 0 else .7,
            nll=.5, wall_seconds=float(arrivals * (1 if condition == "fixed" else 2)), initial_nines=3,
            raw_pi=.025, applied_pi=.025, q=.01, effective_size=100,
            replacement_mass=.975 ** (arrivals // 4), log_replacement_mass=(arrivals // 4) * np.log(.975),
            motion_energy=.02, uncertainty_energy=.01, noise_scale=1., rho=.1,
            boundary_low=False, boundary_high=False, fallback=False,
            mapping_error_ratio=.01, innovation_ratio=1., innovation_log_score=-2.,
            cold_start=False, observer_floor_fraction=.1, measurement_variance_fallback=False,
            covariance_repair_fraction=0., anchor_refresh=False, mapping_error_energy=.01, mapping_error_noise=1.)
        if i == 0:
            for name in ("raw_pi", "applied_pi", "motion_energy", "uncertainty_energy", "noise_scale", "rho", "innovation_log_score"):
                row.pop(name)
        rows.append(row)
    return dict(unit_id=f"{stage}-{schedule}-{condition}-{replica}-{c}-{ridge}", stage=stage,
        replica=replica, schedule=schedule, condition=condition, c=c, ridge=ridge, completed=completed, metrics=rows,
        stream_hash=f"stream-{stage}-{schedule}-{replica}", initial_unit=f"init-{stage}-{replica}", asset_unit=f"asset-{stage}-{replica}",
        summary={"costs": {"evaluation_seconds": 2., "solver_seconds": 3., "controller_seconds": 0. if condition == "fixed" else 2.,
                           "curvature_seconds": 4., "checkpoint_seconds": 1.}, "initial_nines": 3})


def shadow(condition="ema_h16", *, replica=0, schedule="main", completed=True):
    return dict(unit_id=f"shadow-{schedule}-{condition}-{replica}", replica=replica, schedule=schedule,
        condition=condition, completed=completed,
        metrics=[dict(step=1, arrivals=4, segment="low_hold", raw_pi=.01, applied_pi=.01,
                      motion_energy=.1, uncertainty_energy=.2, noise_scale=1., rho=.1,
                      innovation_log_score=-9999., innovation_ratio=1., mapping_error_ratio=.01)],
        summary={"log_score": -2., "regime_log_scores": {"low_hold": -3., "rise": -1.}, "pi_mean": .015})


def snapshot(trajectories=None, shadows=None):
    return dict(schema=1, study_id="scalar-study", status="running", phase="0: acceptance",
        config={"samples_per_step": 4, "dtype": "float32", "seed": 123, "bootstrap_draws": 2000},
        progress={"study_id": "scalar-study", "active_hours": .5, "allocation_hours": 20,
            "active_unit": "next-unit", "step": 3, "total_steps": 20, "heartbeat": "2026-11-01T10:00:00Z", "revision": "v1",
            "storage": {"used_bytes": 1_000_000_000, "external_bytes": 2_000_000_000, "reserved_bytes": 1_000_000_000,
                        "max_bytes": 500_000_000_000, "filesystem_free_bytes": 30_000_000_000}},
        selection={"common_c": .025, "common_ridge": .1, "conditions": list(ARMS), "main_replicas": 8,
                   "stationary_replicas": 4, "fresh_trajectory_target": 48, "frozen_at": "pending", "forecast_fresh_seconds": 3600},
        changes=[], trajectories=trajectories or [], shadows=shadows or [], analysis={})


def write_snapshot(root, value):
    root.mkdir(parents=True, exist_ok=True)
    (root / "progress.json").write_text(json.dumps(value["progress"]))
    (root / "report_snapshot.json").write_text(json.dumps(value))


def comparison(result, condition="ema_h16", control="fixed", group=0):
    return next(c for c in result["groups"][group]["comparisons"] if c["condition"] == condition and c["control"] == control)


def test_five_contrasts_observation_weighting_signs_segments_endpoints_and_costs():
    rows = [trajectory(arm) for arm in ARMS]
    result = analysis.summarize(rows, bootstrap_draws=20, seed=7)
    assert len(result["groups"][0]["comparisons"]) == 5
    assert all(c["control"] in ("fixed", "covariance") for c in result["groups"][0]["comparisons"])
    effect = comparison(result)
    assert effect["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(3.4)
    assert effect["observation_mean_pp"]["old_accuracy"]["mean"] == pytest.approx(-.85)
    assert effect["endpoint_pp"]["nine_recall"]["mean"] == pytest.approx(6)
    assert effect["segments"]["low_hold"]["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(1)
    assert effect["segments"]["rise"]["observation_mean_pp"]["nine_recall"]["mean"] == pytest.approx(4)
    assert effect["costs"]["wall_seconds"]["ratio"]["mean"] == 2
    assert effect["costs"]["wall_seconds"]["difference"]["mean"] == 20
    assert effect["observation_mean_pp"]["nine_recall"]["ci95"] is None
    assert comparison(result, control="covariance")["observation_mean_pp"]["nine_recall"]["mean"] == 0
    json.dumps(result, allow_nan=False)


def test_keep_stage_schedule_c_ridge_and_full_policy_ids_separate():
    ids = ("fixed", "covariance", "ema_h8", "ema_h16", "ema_h32", "kalman_eta0.0001", "kalman_eta0.001", "kalman_eta0.01", "kalman_mean_eta0.001")
    settings = (("development", "main", .025, .1), ("fresh", "main", .025, .1),
                ("fresh", "stationary", .025, .1), ("fresh", "main", .05, .1), ("fresh", "main", .025, .01))
    rows = [trajectory(arm, stage=stage, schedule=schedule, c=c, ridge=r) for stage, schedule, c, r in settings for arm in ids]
    result = analysis.summarize(rows, bootstrap_draws=0)
    assert len(result["groups"]) == 5
    for group in result["groups"]:
        assert set(group["conditions"]) == set(ids)
        assert len(group["comparisons"]) == 15
        assert all(c["n_pairs"] == 1 for c in group["comparisons"])
        assert {r["condition"] for r in group["replicas"]} == set(ids)


def test_complete_pairs_only_prefixes_and_unpaired_arms_explicit():
    rows = [trajectory("fixed"), trajectory("ema_h16"), trajectory("fixed", 1), trajectory("ema_h16", 1, completed=False)]
    result = analysis.summarize(rows)
    effect = comparison(result)
    assert effect["n_pairs"] == 1
    assert effect["unpaired_control_units"] == [rows[2]["unit_id"]]
    assert result["groups"][0]["incomplete_units"] == [rows[3]["unit_id"]]
    assert result["counts"] == {"sealed_trajectories": 4, "complete": 3, "incomplete": 1}
    assert comparison(result, control="covariance")["n_pairs"] == 0


@pytest.mark.parametrize("include_prefixes", (False, True))
def test_required_fresh_block_withholds_completed_pair_without_relabeling_units(include_prefixes):
    rows = [trajectory("fixed"), trajectory("covariance")]
    if include_prefixes:
        rows += [trajectory(arm, completed=False) for arm in ARMS[2:]]
    before = deepcopy(rows)
    legacy = analysis.summarize(rows, bootstrap_draws=0)
    result = analysis.summarize(rows, bootstrap_draws=0, required_conditions=ARMS)
    assert comparison(legacy, condition="covariance")["n_pairs"] == 1
    assert all(c["n_pairs"] == 0 for c in result["groups"][0]["comparisons"])
    group = result["groups"][0]
    expected_units = sorted(t["unit_id"] for t in rows if t["completed"])
    assert group["eligible_replicas"] == []
    assert group["awaiting_full_block_units"] == group["complete_units"] == expected_units
    assert group["replicas"] == legacy["groups"][0]["replicas"]
    assert group["incomplete_units"] == legacy["groups"][0]["incomplete_units"]
    assert result["counts"] == legacy["counts"]
    effect = comparison(result, condition="covariance")
    assert effect["unpaired_control_units"] == [rows[0]["unit_id"]]
    assert effect["unpaired_treatment_units"] == [rows[1]["unit_id"]]
    assert rows == before


def test_required_fresh_blocks_share_eligible_replica_draws_across_all_five_contrasts(monkeypatch):
    rows = [trajectory(arm, r, delta=.01 * (r + 1)) for r in range(4) for arm in ARMS]
    rows += [trajectory("fixed", 4), trajectory("covariance", 4)]
    calls = []
    original = analysis._estimate
    def tracked(values, **kwargs):
        if len(values):
            calls.append((len(values), kwargs["seed"]))
        return original(values, **kwargs)
    monkeypatch.setattr(analysis, "_estimate", tracked)
    result = analysis.summarize(rows, bootstrap_draws=200, seed=9, required_conditions=ARMS)
    group = result["groups"][0]
    assert result["required_conditions"] == list(ARMS)
    assert group["eligible_replicas"] == [0, 1, 2, 3]
    assert len(group["comparisons"]) == 5
    assert all(c["replicas"] == [0, 1, 2, 3] and c["n_pairs"] == 4 for c in group["comparisons"])
    assert len(set(calls)) == 1 and calls[0][0] == 4
    assert group["awaiting_full_block_units"] == sorted(t["unit_id"] for t in rows[-2:])


def test_required_conditions_leave_development_and_legacy_none_output_unchanged():
    rows = [trajectory(arm, stage=stage) for stage in ("development", "fresh") for arm in ("fixed", "covariance")]
    legacy = analysis.summarize(rows, bootstrap_draws=0)
    assert legacy == analysis.summarize(rows, bootstrap_draws=0, required_conditions=None)
    assert "required_conditions" not in legacy
    assert all("eligible_replicas" not in g and "awaiting_full_block_units" not in g for g in legacy["groups"])
    strict = analysis.summarize(rows, bootstrap_draws=0, required_conditions=ARMS)
    assert strict["groups"][0] == legacy["groups"][0]
    assert comparison(strict, condition="covariance", group=0)["n_pairs"] == 1
    assert comparison(strict, condition="covariance", group=1)["n_pairs"] == 0


def test_required_blocks_do_not_complete_across_schedules_stages_or_settings():
    settings = (("fresh", "main", .025, .1), ("fresh", "stationary", .025, .1),
                ("fresh_v2", "main", .025, .1), ("fresh", "main", .05, .1), ("fresh", "main", .025, .01))
    rows = [trajectory(arm, stage=stage, schedule=schedule, c=c, ridge=r)
            for arm, (stage, schedule, c, r) in zip(ARMS, settings)]
    stage, schedule, c, r = settings[-1]
    rows.append(trajectory("fixed", stage=stage, schedule=schedule, c=c, ridge=r))
    result = analysis.summarize(rows, bootstrap_draws=0, required_conditions=ARMS)
    assert all(not g["eligible_replicas"] for g in result["groups"])
    assert all(c["n_pairs"] == 0 for g in result["groups"] for c in g["comparisons"])
    assert sum(len(g["awaiting_full_block_units"]) for g in result["groups"]) == len(rows)


@pytest.mark.parametrize("required", ([], "fixed", ["fixed", "fixed"], ["fixed", None]))
def test_required_conditions_malformed_lists_fail_clearly(required):
    with pytest.raises(ValueError, match="required_conditions"):
        analysis.summarize([], required_conditions=required)


def test_report_validation_recomputes_persisted_block_policy_and_preserves_legacy(tmp_path):
    root = tmp_path / "study"
    rows = [trajectory("fixed"), trajectory("covariance")]
    value = snapshot(rows)
    value["analysis"] = analysis.summarize(rows, bootstrap_draws=0, seed=123, required_conditions=ARMS)
    write_snapshot(root, value)
    _, _, strict, _, _ = report._validated(root)
    assert strict == value["analysis"] and comparison(strict, condition="covariance")["n_pairs"] == 0
    assert "all four retained arms" in report._paired_table(strict).source
    assert "Completed fresh arms awaiting a full block | 2" in report._progress(value, value["progress"], strict).source
    value["analysis"] = analysis.summarize(rows, bootstrap_draws=0, seed=123)
    write_snapshot(root, value)
    _, _, legacy, _, _ = report._validated(root)
    assert legacy == value["analysis"] and comparison(legacy, condition="covariance")["n_pairs"] == 1
    assert "legacy independent complete-pair eligibility" in report._paired_table(legacy).source
    value["analysis"] = {}
    write_snapshot(root, value)
    _, _, rebuilt, _, _ = report._validated(root)
    assert rebuilt["required_conditions"] == list(ARMS)
    assert comparison(rebuilt, condition="covariance")["n_pairs"] == 0


def test_report_block_policy_must_match_selection_and_cannot_bypass_cached_validation(tmp_path):
    root = tmp_path / "study"
    rows = [trajectory(arm) for arm in ARMS]
    value = snapshot(rows)
    value["analysis"] = analysis.summarize(rows, bootstrap_draws=0, required_conditions=ARMS)
    value["selection"]["conditions"] = list(ARMS[:-1])
    write_snapshot(root, value)
    with pytest.raises(ValueError, match="required_conditions.*selection.conditions"):
        report._validated(root)
    value["selection"]["conditions"] = list(reversed(ARMS))
    write_snapshot(root, value)
    assert report._validated(root)[2] == value["analysis"]
    value["analysis"]["groups"][0]["eligible_replicas"] = []
    write_snapshot(root, value)
    with pytest.raises(ValueError, match="corrupt or incompatible"):
        report._validated(root)


def test_whole_replica_bootstrap_reuses_blocks_and_is_order_invariant(monkeypatch):
    rows = [trajectory(arm, r, delta=.01 * (r + 1)) for r in range(4) for arm in ARMS]
    calls = []
    original = analysis._estimate
    def tracked(values, **kwargs):
        calls.append((len(values), kwargs["seed"]))
        return original(values, **kwargs)
    monkeypatch.setattr(analysis, "_estimate", tracked)
    result = analysis.summarize(rows, bootstrap_draws=500, seed=9)
    assert result == analysis.summarize(deepcopy(rows), bootstrap_draws=500, seed=9)
    # Zero-cost fixed controller has no defined cost ratio, hence an empty subset.
    assert len({call for call in calls if call[0]}) == 1 and calls[0][0] == 4
    effect = comparison(result)["observation_mean_pp"]["nine_recall"]
    assert effect["individual"] == pytest.approx([1.7, 3.4, 5.1, 6.8])
    assert effect["mean"] == pytest.approx(4.25)
    assert effect["ci95"][0] < effect["mean"] < effect["ci95"][1]
    assert effect["interval_reliable"]
    reversed_result = analysis.summarize(rows[::-1], bootstrap_draws=500, seed=9)
    assert reversed_result["groups"] == result["groups"]


@pytest.mark.parametrize("name,value", [("arrivals", 99), ("p", .2), ("observed_nines", 19),
    ("segment", "other"), ("stream_hash", "other"), ("initial_unit", "other"), ("asset_unit", "other")])
def test_pair_grid_and_shared_asset_mismatch_fail_without_truncation(name, value):
    rows = [trajectory("fixed"), trajectory("ema_h16")]
    if name in rows[1]:
        rows[1][name] = value
    else:
        rows[1]["metrics"][-1][name] = value
    with pytest.raises(ValueError, match="mismatched"):
        analysis.summarize(rows)


def test_pair_identity_availability_mismatch_rejected():
    rows = [trajectory("fixed"), trajectory("ema_h16")]
    del rows[1]["stream_hash"]
    with pytest.raises(ValueError, match="availability"):
        analysis.summarize(rows)


def test_fallback_reason_is_str_or_bool_and_converts_only_for_plotting(tmp_path, captured_plots):
    rows = [trajectory(arm) for arm in ARMS]
    for t in rows:
        t['metrics'][0]['fallback'] = ''
        t['metrics'][1]['fallback'] = 'unsupported_movement_or_noise'
    value = snapshot(rows)
    before = deepcopy(value)
    result = analysis.summarize(rows, bootstrap_draws=0)
    report._policy(value, result, tmp_path / 'fallback.png')
    fallback_curves = captured_plots[-1][5]['lines'][2::3]
    assert len(fallback_curves) == 4
    for line in fallback_curves:
        np.testing.assert_array_equal(line['y'], [0, 1, 0, 0])
    assert value == before
    rows[0]['metrics'][1]['fallback'] = 1
    with pytest.raises(ValueError, match='fallback'):
        analysis.summarize(rows)


def test_variable_replica_horizons_keep_separate_grid_cohorts_and_statistics(tmp_path, captured_plots):
    rows = [trajectory(arm, r) for r in (0, 1, 2) for arm in ARMS]
    for t in rows:
        if t['replica'] != 1:
            continue
        t['schedule_factor'] = 2.
        for row in t['metrics']:
            for field in ('arrivals', 'step', 'expected_nines', 'observed_nines', 'wall_seconds'):
                row[field] *= 2
            row['nine_precision'] = None if row['arrivals'] == 24 else row['nine_precision']
    value = snapshot(rows)
    result = analysis.summarize(rows, bootstrap_draws=200)
    effect = comparison(result)
    assert effect['n_pairs'] == 3
    assert effect['observation_mean_pp']['nine_recall']['individual'] == pytest.approx([3.4] * 3)
    report._fresh_accuracy(value, result, tmp_path / 'variable.png')
    main = captured_plots[-1][0]
    paths = [line for line in main['lines'] if not line['label'].startswith('Frozen')]
    assert len(paths) == 8
    assert {line['label'].split('horizon=')[1].split(' ')[0] for line in paths} == {'20', '40'}
    assert {tuple(line['x']) for line in paths} == {(0, 4, 12, 20), (0, 8, 24, 40)}
    assert '40-digit reference' in main['title']
    report._precision_exposure(value, result, tmp_path / 'variable-precision.png')
    precision = captured_plots[-1][0]
    assert len(precision['lines']) == 8
    long_curves = [line for line in precision['lines'] if line['x'][-1] == 40]
    assert len(long_curves) == 4 and all(np.isnan(line['y'][2]) for line in long_curves)
    report._policy(value, result, tmp_path / 'variable-policy.png')
    assert {tuple(line['x']) for line in captured_plots[-1][1]['lines']} == {(0, 4, 12, 20), (0, 8, 24, 40)}


@pytest.mark.parametrize("field,value", [("applied_pi", 2), ("motion_energy", -.1), ("noise_scale", float("nan")),
    ("q", -1), ("innovation_log_score", float("inf")), ("log_replacement_mass", 1), ("nine_recall", 30)])
def test_corrupt_scalars_are_rejected(field, value):
    row = trajectory()
    row["metrics"][-1][field] = value
    with pytest.raises(ValueError):
        analysis.summarize([row])


def test_bootstrap_draw_bounds_and_validation():
    assert analysis.summarize([], bootstrap_draws=10**9)["bootstrap"]["draws"] == analysis.MAX_BOOTSTRAP_DRAWS
    assert analysis.summarize([], bootstrap_draws=-1)["bootstrap"]["draws"] == 0
    with pytest.raises(ValueError, match="seed"):
        analysis.summarize([], seed=True)
    with pytest.raises(ValueError, match="bootstrap_draws"):
        analysis.summarize([], bootstrap_draws=1.5)


@pytest.fixture
def captured_plots(monkeypatch):
    recorded = []
    original = report._figure
    @contextmanager
    def capture(*args, **kwargs):
        with original(*args, **kwargs) as (figure, axes):
            yield figure, axes
            recorded.append([dict(title=ax.get_title(), xlabel=ax.get_xlabel(), ylim=ax.get_ylim(), bands=len(ax.patches),
                legend=ax.get_legend() is not None, texts=[t.get_text() for t in ax.texts],
                lines=[dict(label=line.get_label(), x=np.asarray(line.get_xdata()).copy(), y=np.asarray(line.get_ydata()).copy()) for line in ax.lines],
                points=[np.asarray(collection.get_offsets()).copy() for collection in ax.collections]) for ax in axes.flat])
    monkeypatch.setattr(report, "_figure", capture)
    return recorded


def test_shadow_calibration_uses_only_sealed_summary_not_prefix_or_metric_log_scores(tmp_path, captured_plots):
    complete = shadow()
    incomplete = shadow("kalman_eta0.001", completed=False)
    incomplete["summary"]["log_score"] = -999.
    value = snapshot(shadows=[complete, incomplete])
    before = deepcopy(value)
    report._shadow_calibration(value, analysis.summarize([]), tmp_path / "shadow.png")
    axes = captured_plots[-1]
    assert len(axes[0]["points"]) == 1
    assert axes[0]["points"][0][0, 0] == -2
    np.testing.assert_array_equal(axes[1]["lines"][0]["y"], [-3, -1])
    assert "regime-balanced" in axes[0]["title"]
    assert value == before


def test_completed_shadow_summary_corruption_rejected_but_empty_prefix_valid():
    incomplete = shadow(completed=False)
    incomplete.update(summary={}, metrics=[])
    analysis.validate_shadows([incomplete])
    complete = shadow()
    del complete["summary"]["log_score"]
    with pytest.raises(ValueError, match="log_score"):
        analysis.validate_shadows([complete])


def test_shadow_regimes_follow_trajectory_not_alphabetical_summary(tmp_path, captured_plots):
    complete = shadow()
    order = ["low_hold_initial", "rise", "high_hold", "return", "low_hold_final"]
    complete["metrics"] = [dict(segment=segment) for segment in order]
    complete["summary"]["regime_log_scores"] = {segment: order.index(segment) for segment in sorted(order)}
    report._shadow_calibration(snapshot(shadows=[complete]), analysis.summarize([]), tmp_path / "shadow.png")
    line = captured_plots[-1][1]["lines"][0]
    np.testing.assert_array_equal(line["x"], order)
    np.testing.assert_array_equal(line["y"], range(5))


def test_fresh_accuracy_is_2x2_main_stationary_complete_pairs_with_frozen_reference(tmp_path, captured_plots):
    rows = [trajectory(arm, schedule=schedule) for schedule in ("main", "stationary") for arm in ARMS]
    rows.append(trajectory("kalman_mean_eta0.001", 4, completed=False))
    value = snapshot(rows)
    report._fresh_accuracy(value, analysis.summarize(rows, bootstrap_draws=0), tmp_path / "fresh.png")
    axes = captured_plots[-1]
    assert len(axes) == 4
    assert [a["title"].split(":")[0] for a in axes] == ["Main", "Main", "Stationary", "Stationary"]
    for ax in axes:
        assert ax["xlabel"] == "Online digits observed" and ax["bands"] == 2
        assert len(ax["lines"]) == 5
        assert any(line["label"].startswith("Frozen initial reference") for line in ax["lines"])
        assert not any("Kalman mean" in line["label"] for line in ax["lines"])


def test_frozen_reference_matches_each_length_cohort(tmp_path, captured_plots):
    rows = [trajectory(arm, replica) for replica in (0, 1) for arm in ARMS]
    for t in rows:
        if t["replica"] == 1:
            for row in t["metrics"]:
                row["arrivals"] *= 2
                row["frozen_old_accuracy"] = .7
    report._fresh_accuracy(snapshot(rows), analysis.summarize(rows, bootstrap_draws=0), tmp_path / "cohorts.png")
    ax = captured_plots[-1][1]
    references = [line for line in ax["lines"] if line["label"].startswith("Frozen initial reference")]
    assert len(references) == 2
    assert {tuple(line["x"]) for line in references} == {(0, 20), (0, 40)}
    assert {tuple(line["y"]) for line in references} == {(90., 90.), (70., 70.)}
    assert not captured_plots[-1][0]["legend"]
    assert ax["legend"]


def test_paired_effect_figure_focuses_on_full_path_not_endpoints(tmp_path, captured_plots):
    rows = [trajectory(arm) for arm in ARMS]
    result = analysis.summarize(rows, bootstrap_draws=0)
    report._paired_cost(snapshot(rows), result, tmp_path / "effects.png")
    expected = [c["observation_mean_pp"]["nine_recall"]["mean"] for c in result["groups"][0]["comparisons"]]
    points = [line["x"][0] for line in captured_plots[-1][0]["lines"] if len(line["x"]) == 1]
    np.testing.assert_allclose(points, expected)
    assert "Full-path mean difference" in captured_plots[-1][0]["xlabel"]
    assert len(captured_plots[-1]) == 6
    kalman = comparison(result, condition="kalman_eta0.001")
    expected_regimes = [kalman["segments"][key]["observation_mean_pp"]["nine_recall"]["mean"] for key in ("rise", "low_hold")]
    regime_points = [line["x"][0] for line in captured_plots[-1][2]["lines"] if len(line["x"]) == 1]
    np.testing.assert_allclose(regime_points, expected_regimes)


def test_conclusion_separates_schedules_and_reads_uncertainty_without_equivalence():
    rows = [trajectory(arm, replica, schedule=schedule)
            for schedule in ("main", "stationary") for replica in range(4) for arm in ARMS]
    result = analysis.summarize(rows, bootstrap_draws=20)
    prose = report._conclusion(result).source
    assert "Changing | Kalman" in prose and "Stationary | Kalman" in prose
    assert "higher 9 recall; lower 0--8 retention" in prose
    assert "not deployment FLOP estimates" in prose
    for group in result["groups"]:
        for contrast in group["comparisons"]:
            contrast["observation_mean_pp"]["nine_recall"]["ci95"] = [-1., 1.]
    assert "9 recall difference remains unresolved" in report._conclusion_reading(result)
    assert "does not prove preservation" in report._conclusion(result).source

    for group in result["groups"]:
        for contrast in group["comparisons"]:
            if contrast["condition"].startswith("kalman_") and contrast["control"] == "covariance":
                for metric in ("nine_recall", "old_accuracy"):
                    contrast["observation_mean_pp"][metric]["ci95"] = [.1, .2]
    assert "Kalman improves both accuracies" in report._conclusion_reading(result)
    assert "does not isolate genuine-motion detection" in report._conclusion_reading(result)


def test_precision_keeps_undefined_cohort_gaps_without_nanmean_replacement(tmp_path, captured_plots):
    rows = [trajectory(arm, r) for r in (0, 1) for arm in ("fixed", "ema_h16")]
    for t in rows:
        t["metrics"][0]["nine_precision"] = .6
        if t["replica"] == 1:
            t["metrics"][1]["nine_precision"] = None
    report._precision_exposure(snapshot(rows), analysis.summarize(rows, bootstrap_draws=0), tmp_path / "precision.png")
    main = captured_plots[-1][0]
    assert len(main["lines"]) == 2
    for line in main["lines"]:
        assert np.isnan(line["y"][1]) and np.isfinite(line["y"][[0, 2, 3]]).all()
        np.testing.assert_array_equal(line["x"], [0, 4, 12, 20])
    for t in rows:
        for row in t["metrics"]:
            row["nine_precision"] = None
    report._precision_exposure(snapshot(rows), analysis.summarize(rows, bootstrap_draws=0), tmp_path / "undefined.png")
    main = captured_plots[-1][0]
    assert not main["lines"] and not main["legend"]
    assert "No complete-cohort precision path" in main["texts"][0]
    assert "0/4 defined" in main["texts"][1]


def test_development_selection_and_policy_panels_do_not_mix_settings_or_stages(tmp_path, captured_plots):
    rows = [trajectory(arm, stage="development", c=c) for c in (.025, .05) for arm in ARMS]
    value = snapshot(rows)
    result = analysis.summarize(rows, bootstrap_draws=0)
    before = deepcopy(value), deepcopy(result)
    report._development(value, result, tmp_path / "dev.png")
    assert all(len(ax["lines"]) == 4 for ax in captured_plots[-1][:2])
    assert all("c=0.025" in line["label"] for line in captured_plots[-1][0]["lines"])
    assert sum(len(points) for points in captured_plots[-1][2]["points"]) == 8
    report._policy(value, result, tmp_path / "policy.png")
    assert len(captured_plots[-1]) == 6
    assert all("development" in line["label"] and "c=0.025" in line["label"] for ax in captured_plots[-1] for line in ax["lines"])
    assert all(captured_plots[-1][i]["ylim"][0] == 0 for i in (0, 3, 4, 5))
    assert captured_plots[-1][5]["ylim"][1] == 1
    assert before == (value, result)


def test_conclusion_separates_fresh_schedules_overhead_percent_and_covariance_summary():
    rows = [trajectory(arm, r) for r in range(4) for arm in ARMS]
    rows += [trajectory(arm, stage="development", delta=.1) for arm in ARMS]
    rows += [trajectory(arm, schedule="stationary", delta=.15) for arm in ARMS]
    result = analysis.summarize(rows, bootstrap_draws=200)
    source = report._conclusion(result).source
    assert "Fresh policies versus the tuned fixed control" in source
    assert "| Changing | EMA H=16 (n=4) | +3.40 [3.40, 3.40] | -0.85 [-0.85, -0.85] | 100.0% |" in source
    assert "| Stationary | EMA H=16 (n=1) |" in source
    assert "one-time observer calibration" in source and "shared initial learner/Fisher costs are excluded" in source
    assert "trade-off" in source and "does not prove preservation" in source
    table = report._paired_table(result).source
    assert "EMA H=16 - Covariance only" in table
    assert "Kalman eta=0.001 - Covariance only" in table
    assert "fresh / stationary" in table and "development /" not in table
    assert "analysis.json" in table and "repeated inspection" in table


def test_conclusion_does_not_relabel_stationary_or_development_as_changing_evidence():
    rows = [trajectory(arm, stage=stage, schedule=schedule) for stage, schedule in (("development", "main"), ("fresh", "stationary")) for arm in ARMS]
    summary = report._conclusion_summary(analysis.summarize(rows, bootstrap_draws=0))
    assert "| Stationary |" in summary
    assert "| Changing |" not in summary and "development" not in summary
    assert report._conclusion_summary(analysis.summarize(rows[:4], bootstrap_draws=0)) == ""


def test_progress_settings_remain_compact_and_count_complete_retained_blocks():
    rows = [trajectory(arm) for arm in ARMS]
    rows += [trajectory("fixed", replica=1), trajectory("ema_h16", replica=1, completed=False)]
    value = snapshot(rows)
    value["selection"]["candidates"] = [{"huge_unused_candidate": "detail" * 1000}] * 100
    value["changes"] = [{"revision": "v2", "change": "Repair measurement scaling. " + "Detailed validation. " * 100}]
    result = analysis.summarize(rows, bootstrap_draws=0)
    settings = report._settings(value, result).source
    assert "integrate over model labels instead" in settings
    assert "not the Kalman filter's covariance" in settings
    assert len(settings) < 5000 and "huge_unused_candidate" not in settings
    assert "Repair measurement scaling." in settings and "Detailed validation" not in settings
    for phrase in ("1.00 h", "main=8; stationary=4", "c=0.025; ridge=0.1", "pure NLL", "p_ref=.1", "independent stratified holdouts"):
        assert phrase in settings
    progress = report._progress(value, value["progress"], result).source
    for phrase in ("1 / 8 blocks; 5 / 32 trajectories", "3/20", "4.00 / 500.00 / 496.00", "0.50 / 20.00"):
        assert phrase in progress


def test_publishes_six_nonblank_pngs_portable_outputs_and_no_scientific_writes(tmp_path):
    root, notebook = tmp_path / "study", tmp_path / "plan17_results.ipynb"
    rows = [trajectory(arm, stage=stage, schedule=schedule) for stage in ("development", "fresh") for schedule in ("main", "stationary") for arm in ARMS]
    value = snapshot(rows, [shadow()])
    value["analysis"] = analysis.summarize(rows, bootstrap_draws=20, seed=123)
    write_snapshot(root, value)
    sealed = root / "units" / "sealed"
    sealed.mkdir(parents=True)
    (sealed / "COMPLETED").write_text("immutable")
    (sealed / "checkpoint.pt").write_bytes(b"not a model; never load this")
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    manifest = report.build_report(root, notebook)
    assert manifest == json.loads((root / "report" / "latest.json").read_text())
    book = nbformat.read(notebook, as_version=4)
    nbformat.validate(book)
    assert book.cells[1].source.startswith("## Progress")
    assert "Phase 5 follow-up" in book.cells[1].source
    assert book.cells[-1].id == "plan17-retention-followup"
    assert "Drafted for review; not started" in book.cells[-1].source
    assert "No Phase 5\nresults are available yet" in book.cells[-1].source
    assert any(c.id == "plan17-conclusion" for c in book.cells[:-1])
    figures = [c for c in book.cells if c.cell_type == "code"]
    assert len(figures) == 6
    for cell in figures:
        assert "Image(filename=" in cell.source and len(cell.outputs) == 1
        data = base64.b64decode(cell.outputs[0].data["image/png"])
        picture = Image.open(io.BytesIO(data)).convert("RGB")
        assert picture.width > 1000 and picture.height > 600
        assert np.asarray(picture).std() > 10
        assert (Path(manifest["report_dir"]) / (cell.id.removeprefix("plan17-") + ".png")).read_bytes() == data
    assert [str(p) for p in root.rglob("checkpoint.pt")] == [str(sealed / "checkpoint.pt")]
    assert all(Path(path).read_bytes() == data for path, data in before.items())


def test_pending_notebook_and_configured_bootstrap_seed_with_no_trajectories(tmp_path):
    root, notebook = tmp_path / "pending", tmp_path / "pending.ipynb"
    value = snapshot()
    value["selection"] = {}
    value["config"]["bootstrap_seed"] = 456
    write_snapshot(root, value)
    manifest = report.build_report(root, notebook)
    prose = "\n".join(c.source for c in nbformat.read(notebook, as_version=4).cells if c.cell_type == "markdown")
    assert "No complete fresh main paired evidence yet" in prose
    assert "Partial / incomplete" in prose
    assert manifest["counts"]["complete"] == 0
    saved = json.loads((Path(manifest["report_dir"]) / "analysis.json").read_text())
    assert saved["bootstrap"]["seed"] == 456 and saved["bootstrap"]["draws"] == 2000


def test_historical_publisher_cannot_replace_combined_phase5_notebook(tmp_path):
    root, notebook = tmp_path / "historical", tmp_path / "plan17_results.ipynb"
    write_snapshot(root, snapshot())
    book = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("Phase 5 evidence")],
        metadata={"plan17_retention_report": {"schema": 1, "study_id": "retention_v1"}})
    nbformat.write(book, notebook)
    before = notebook.read_bytes()
    with pytest.raises(ValueError, match="Phase 5 owns"):
        report.build_report(root, notebook)
    assert notebook.read_bytes() == before
    assert not (root / "report").exists()


def test_report_generation_tracks_shared_helper_source(tmp_path, monkeypatch):
    root, notebook = tmp_path / "study", tmp_path / "results.ipynb"
    write_snapshot(root, snapshot())
    first = report.build_report(root, notebook)
    assert report.build_report(root, notebook)["report_dir"] == first["report_dir"]
    original = Path.read_bytes
    helper = Path(report.__file__).parent.parent / "diagonal_lfu" / "report.py"

    def changed(path):
        return original(path) + (b"\n# reporting version probe\n" if path == helper else b"")

    monkeypatch.setattr(Path, "read_bytes", changed)
    second = report.build_report(root, notebook)
    assert second["report_dir"] != first["report_dir"]
    assert Path(first["report_dir"]).is_dir()


def test_smoke_banner_and_no_production_conclusion(tmp_path):
    root, notebook = tmp_path / "smoke", tmp_path / "smoke.ipynb"
    value = snapshot([trajectory(arm) for arm in ARMS])
    value["config"]["smoke"] = True
    value["status"] = "completed"
    write_snapshot(root, value)
    report.build_report(root, notebook)
    book = nbformat.read(notebook, as_version=4)
    prose = "\n".join(c.source for c in book.cells if c.cell_type == "markdown")
    assert "SMOKE ACCEPTANCE ONLY / NOT PRODUCTION EVIDENCE" in book.cells[0].source
    assert "## Smoke Accuracy" in prose and "## Fresh Accuracy" not in prose
    assert "## Smoke Acceptance Conclusion" in prose and "Fresh main versus fixed" not in prose
    assert not any(c.id == "plan17-retention-followup" for c in book.cells)
    assert "Phase 5 follow-up" not in book.cells[1].source
    assert book.metadata.plan17_report.smoke is True


@pytest.mark.parametrize("corruption", ("schema", "missing", "stale", "wrong_kind", "altered_effect", "nonfinite", "overflow", "study_id", "shadow"))
def test_corrupt_artifacts_fail_before_any_publication(tmp_path, corruption):
    root, notebook = tmp_path / "corrupt", tmp_path / "report.ipynb"
    value = snapshot([trajectory(arm) for arm in ARMS], [shadow()])
    value["analysis"] = analysis.summarize(value["trajectories"], bootstrap_draws=20)
    if corruption == "schema":
        value["schema"] = 2
    elif corruption == "missing":
        del value["shadows"]
    elif corruption == "stale":
        value["analysis"] = analysis.summarize([])
    elif corruption == "wrong_kind":
        value["analysis"]["kind"] = "plan16-paired-analysis"
    elif corruption == "altered_effect":
        comparison(value["analysis"])["observation_mean_pp"]["nine_recall"]["mean"] = 999
    elif corruption == "nonfinite":
        value["trajectories"][0]["metrics"][0]["noise_scale"] = float("nan")
    elif corruption == "overflow":
        value["config"]["unused_scalar"] = float("inf")
    elif corruption == "study_id":
        value["progress"]["study_id"] = "wrong"
    else:
        del value["shadows"][0]["summary"]["log_score"]
    write_snapshot(root, value)
    with pytest.raises(ValueError):
        report.build_report(root, notebook)
    assert not notebook.exists() and not (root / "report").exists()


def test_failed_refresh_preserves_prior_report_and_cleans_staging(tmp_path, monkeypatch):
    root, notebook = tmp_path / "study", tmp_path / "report.ipynb"
    value = snapshot()
    write_snapshot(root, value)
    report.build_report(root, notebook)
    before = notebook.read_bytes(), (root / "report" / "latest.json").read_bytes()
    def fail(*args):
        raise RuntimeError("synthetic plot failure")
    monkeypatch.setattr(report, "_development", fail)
    with pytest.raises(RuntimeError, match="synthetic plot failure"):
        report.build_report(root, notebook)
    assert before == (notebook.read_bytes(), (root / "report" / "latest.json").read_bytes())
    assert not list((root / "report").glob(".building-*"))


def test_missing_artifacts_fail_and_imports_do_not_load_training_or_torch(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.build_report(tmp_path, tmp_path / "report.ipynb")
    subprocess.run([sys.executable, "-c", "import sys; from mnist_experiment.kalman_pi import report; "
        "assert 'torch' not in sys.modules; "
        "assert 'mnist_experiment.kalman_pi.run' not in sys.modules; "
        "assert 'mnist_experiment.diagonal_lfu.data' not in sys.modules; "
        "assert 'mnist_experiment.diagonal_lfu.run' not in sys.modules"], check=True)
