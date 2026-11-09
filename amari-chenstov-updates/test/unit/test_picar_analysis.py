"""Scalar synthetic fixtures, not reported experimental outcomes."""

import numpy as np
import pytest

from numerical_experiment.picar import analysis


def curves():
    grid = [0, 10, 20, 30]
    control = analysis.LearningCurve(grid, [0, .4, .8, .9], [0, .01, .02, .03], grid)
    fd = analysis.LearningCurve(grid, [0, .8, .9, .95], [0, .01, .02, .03], [0, 40, 80, 120])
    return control, fd


def test_failure_and_retention_reference_fixtures():
    assert analysis.budget_averaged_failure([0, 1, 2], [0, .5, 1]) == .5
    result = analysis.retention_summary([0, 1, 2], [.8, .7, .6], initial_success=.8)
    assert result["average_failure_increase"] == pytest.approx(.1)
    assert result["endpoint_failure_increase"] == pytest.approx(.2)
    assert analysis.efficiency_metrics(100, 50, 10, 20) == {"f": .5, "c": 2., "S": .25}


@pytest.mark.parametrize("samples,success", [([1, 2], [.1, .2]), ([0, 0], [.1, .2]),
    ([0, 1], [.1, 1.2]), ([0], [.1]), ([0, 1], [.1, float("nan")])])
def test_bad_curve_designs_fail(samples, success):
    with pytest.raises(ValueError):
        analysis.budget_averaged_failure(samples, success)


def test_paired_mean_ci_uses_complete_pairs_and_explicit_missing_ids():
    result = analysis.paired_mean_ci({0: 3., 1: 5.}, {0: 2., 1: 3.}, draws=1000, seed=9)
    assert result["mean"] == 1.5 and result["ci95"] == [1., 2.]
    assert result["n_pairs"] == 2 and result["paired_differences"] == [1., 2.]
    assert result == analysis.paired_mean_ci({0: 3., 1: 5.}, {0: 2., 1: 3.}, draws=1000, seed=9)
    with pytest.raises(ValueError, match="incomplete"):
        analysis.paired_mean_ci({0: 3., 1: 5.}, {0: 2.})
    incomplete = analysis.paired_mean_ci({0: 3., 1: 5.}, {0: 2.}, allow_incomplete=True)
    assert incomplete["missing_fd"] == [1] and incomplete["ci95"] is None
    empty = analysis.paired_mean_ci({}, {})
    assert empty["mean"] is None and empty["n_pairs"] == 0


def test_measured_confirmation_and_interpolated_crossing_are_distinct():
    control, fd = curves()
    crossing = analysis.first_supported_crossing(control.samples, control.success,
                                                 control.retention_increase)
    assert crossing.status == "attained"
    assert (crossing.measured_samples, crossing.confirmation_samples) == (20., 30.)
    assert (crossing.lower_samples, crossing.upper_samples) == (10., 20.)
    assert crossing.interpolated_samples == pytest.approx(17.5)
    result = analysis.sample_efficiency(control, fd)
    assert result["status"] == "supported"
    assert result["f"] == pytest.approx(.5)
    assert result["c"] == pytest.approx(2.)
    assert result["S"] == pytest.approx(.25)


def test_retention_gate_and_unconfirmed_last_checkpoint_do_not_fabricate_crossings():
    grid = [0, 10, 20, 30]
    gated = analysis.first_supported_crossing(grid, [0, .8, .9, .95], [0, .2, 0, 0])
    assert gated.status == "attained" and gated.measured_samples == 20
    assert gated.interpolated_samples is None
    censored = analysis.first_supported_crossing(grid, [0, .1, .4, .8], [0] * 4)
    assert censored.status == "right_censored" and censored.censored_at == 30
    windowed = analysis.first_supported_crossing(grid, [0, .1, .8, .9], [0] * 4, max_samples=20)
    assert windowed.status == "right_censored" and windowed.censored_at == 20


def test_initial_attainment_never_divides_by_zero_new_samples():
    initial = analysis.LearningCurve([0, 10], [.8, .9], [0, 0], [1, 2])
    assert analysis.sample_efficiency(initial, initial)["status"] == "unavailable"
    with pytest.raises(ValueError):
        analysis.efficiency_metrics(0, 0, 1, 1)
    with pytest.raises(ValueError):
        analysis.efficiency_metrics(100, 50, 1, 0)


def test_whole_pair_bootstrap_keeps_failed_draws_and_reports_conditional_intervals():
    control, good = curves()
    poor = analysis.LearningCurve(good.samples, [0, .1, .2, .3], [0] * 4, good.operational_seconds)
    result = analysis.bootstrap_sample_efficiency([(control, good), (control, poor)], draws=200, seed=12)
    assert 0 < result["supported_draws"] < 200
    assert result["censored_or_unsupported_draws"] + result["supported_draws"] == 200
    assert sum(result["draw_status_counts"].values()) == 200
    assert result["attainment_fraction"] == {"control": 1., "fd": .5}
    assert result["conditional_ci95"] is not None
    assert result == analysis.bootstrap_sample_efficiency([(control, good), (control, poor)], draws=200, seed=12)
    single = analysis.bootstrap_sample_efficiency([(control, good)], draws=3)
    assert single["conditional_ci95"] is None


def test_bootstrap_rejects_unmatched_checkpoint_grids():
    control, fd = curves()
    other = analysis.LearningCurve([0, 11, 20, 30], fd.success, fd.retention_increase, fd.operational_seconds)
    with pytest.raises(ValueError, match="common measured"):
        analysis.bootstrap_sample_efficiency([(control, other)], draws=2)
