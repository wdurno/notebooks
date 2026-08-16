import pytest

from src.results_analysis import (
    AnalysisArtifactError,
    phase9_durable_crossing_rows,
    phase9_equivalent_data_multiplier_rows,
    phase9_expected_trajectory_rows,
    phase9_exposure_auc_rows,
    phase9_fixed_budget_rows,
    phase9_metric_availability_rows,
    phase9_joint_durable_crossing_rows,
    phase9_paired_dominance_bootstrap_rows,
    phase9_paired_auc_rows,
    phase9_paired_fixed_budget_rows,
    require_phase9_metrics,
)


def _trajectory_rows() -> list[dict]:
    rows = []
    curves = {
        "control": ("control", None, [0.40, 0.75, 0.86, 0.89]),
        "treatment": (
            "treatment",
            "control-run-{replica}",
            [0.50, 0.91, 0.88, 0.95],
        ),
    }
    for replica in (1, 2):
        offset = -0.01 if replica == 1 else 0.01
        for cell, (kind, control_template, values) in curves.items():
            run_id = f"{cell}-run-{replica}"
            control_run_id = (
                None
                if control_template is None
                else control_template.format(replica=replica)
            )
            for step, value in enumerate(values):
                observations = 2 * (step + 1)
                rows.append(
                    {
                        "profile": "plan2",
                        "cell": cell,
                        "kind": kind,
                        "method": "rank8",
                        "run_id": run_id,
                        "control_run_id": control_run_id,
                        "replica_id": f"replica-{replica}",
                        "replica_index": replica,
                        "step": step,
                        "p": step / 3,
                        "samples_per_step": 2,
                        "after_nine_ovr_accuracy": value + offset,
                        "after_nine_precision": value + offset - 0.05,
                        "after_nine_recall": value + offset + 0.05,
                        "after_non_nine_accuracy": 0.9 - 0.02 * step,
                        "after_environment_accuracy": 0.8 + 0.02 * step,
                        "after_cumulative_observations": observations,
                        "after_cumulative_nine_observations": step + 1,
                        "after_cumulative_non_nine_observations": step + 1,
                        "after_cumulative_unique_observations": observations - 1,
                        "after_cumulative_unique_nine_observations": step + 1,
                        "after_cumulative_unique_non_nine_observations": step,
                    }
                )
    return rows


def test_expected_trajectory_and_durable_crossing_handle_nonmonotonicity() -> None:
    expected = phase9_expected_trajectory_rows(
        _trajectory_rows(), metrics=("after_nine_ovr_accuracy",)
    )
    treatment = [
        row for row in expected if row["cell"] == "treatment"
    ]

    assert len(treatment) == 4
    assert treatment[1]["mean"] == pytest.approx(0.91)
    assert treatment[1]["standard_deviation"] is not None
    assert treatment[1]["metric_complete"] is True
    crossings = {
        row["cell"]: row
        for row in phase9_durable_crossing_rows(expected, threshold=0.90)
    }
    assert crossings["treatment"]["crossed"] is True
    assert crossings["treatment"]["step"] == 3
    assert crossings["treatment"][
        "mean_after_cumulative_observations"
    ] == 8
    assert crossings["control"]["crossed"] is False
    assert crossings["control"]["step"] is None

    joint = {
        row["cell"]: row
        for row in phase9_joint_durable_crossing_rows(
            phase9_expected_trajectory_rows(
                _trajectory_rows(),
                metrics=(
                    "after_nine_ovr_accuracy",
                    "after_nine_precision",
                    "after_nine_recall",
                    "after_environment_accuracy",
                ),
            ),
            thresholds={
                "after_nine_ovr_accuracy": 0.90,
                "after_nine_precision": 0.85,
                "after_nine_recall": 0.95,
                "after_environment_accuracy": 0.85,
            },
        )
    }
    assert joint["treatment"]["crossed"] is True
    assert joint["treatment"]["step"] == 3
    assert joint["control"]["crossed"] is False


def _edm_expected_rows(*, unmatched: bool = False) -> list[dict]:
    metrics = (
        "after_nine_ovr_accuracy",
        "after_nine_precision",
        "after_nine_recall",
        "after_environment_accuracy",
    )
    values = {
        ("treatment", 2): 0.80 if not unmatched else 0.95,
        ("control", 2): 0.60,
        ("control", 4): 0.75,
        ("control", 8): 0.85,
    }
    rows = []
    for (cell, samples_per_step), value in values.items():
        for metric in metrics:
            rows.append(
                {
                    "profile": "plan2",
                    "cell": cell,
                    "method": "rank8",
                    "samples_per_step": samples_per_step,
                    "step": 49,
                    "p": 49 / 99,
                    "metric": metric,
                    "mean": value,
                    "metric_complete": True,
                    "replica_count": 5,
                    "mean_after_cumulative_unique_nine_observations": (
                        12.5 * samples_per_step
                    ),
                }
            )
    return rows


def test_equivalent_data_multiplier_reports_discrete_joint_bracket() -> None:
    rows = phase9_equivalent_data_multiplier_rows(
        _edm_expected_rows(),
        baseline_cell="control",
        treatment_cells=("treatment",),
    )
    joint = next(row for row in rows if row["comparison"] == "joint_primary")

    assert joint["status"] == "bracketed"
    assert joint["baseline_samples_per_step_lower"] == 4
    assert joint["baseline_samples_per_step_upper"] == 8
    assert joint["multiplier_lower"] == pytest.approx(2.0)
    assert joint["multiplier_upper"] == pytest.approx(4.0)
    assert joint["minimum_replica_count"] == 5


def test_equivalent_data_multiplier_preserves_right_censoring() -> None:
    rows = phase9_equivalent_data_multiplier_rows(
        _edm_expected_rows(unmatched=True),
        baseline_cell="control",
        treatment_cells=("treatment",),
    )
    joint = next(row for row in rows if row["comparison"] == "joint_primary")

    assert joint["status"] == "right_censored"
    assert joint["multiplier_lower"] == pytest.approx(4.0)
    assert joint["multiplier_upper"] is None


def test_paired_dominance_bootstrap_uses_joint_replica_resamples() -> None:
    metrics = (
        "after_nine_ovr_accuracy",
        "after_nine_precision",
        "after_nine_recall",
        "after_environment_accuracy",
    )
    rows = []
    for replica in range(1, 6):
        offset = (replica - 3) * 0.01
        for cell, samples_per_step, value in (
            ("treatment", 2, 0.80),
            ("control", 4, 0.75),
            ("control", 8, 0.85),
        ):
            rows.append(
                {
                    "profile": "plan2",
                    "cell": cell,
                    "method": "rank8",
                    "samples_per_step": samples_per_step,
                    "step": 49,
                    "p": 49 / 99,
                    "replica_index": replica,
                    "after_cumulative_unique_nine_observations": (
                        12.5 * samples_per_step
                    ),
                    **{metric: value + offset for metric in metrics},
                }
            )

    results = phase9_paired_dominance_bootstrap_rows(
        rows,
        baseline_cell="control",
        treatment_cells=("treatment",),
        treatment_samples_per_step=2,
        baseline_samples_per_step=(4, 8),
        steps=(49,),
    )
    indexed = {row["baseline_samples_per_step"]: row for row in results}

    assert indexed[4]["bootstrap_mode"] == "exact"
    assert indexed[4]["bootstrap_draw_count"] == 5**5
    assert indexed[4]["joint_match_probability"] == 0.0
    assert indexed[8]["joint_match_probability"] == 1.0
    assert indexed[8]["exposure_multiplier"] == pytest.approx(4.0)


def test_auc_and_fixed_budget_contrasts_remain_paired_by_replica() -> None:
    rows = _trajectory_rows()
    auc = phase9_exposure_auc_rows(
        rows, metrics=("after_nine_ovr_accuracy",)
    )
    auc_metric = "after_nine_ovr_accuracy_observation_auc"
    auc_pairs = phase9_paired_auc_rows(auc, metrics=(auc_metric,))

    assert len(auc_pairs) == 2
    assert all(row[f"delta_{auc_metric}"] > 0.0 for row in auc_pairs)

    fixed = phase9_fixed_budget_rows(
        rows,
        observation_budgets=(4,),
        metrics=("after_nine_ovr_accuracy",),
    )
    fixed_pairs = phase9_paired_fixed_budget_rows(
        fixed, metrics=("after_nine_ovr_accuracy",)
    )
    assert len(fixed_pairs) == 2
    assert {row["step"] for row in fixed} == {1}
    assert all(
        row["delta_after_nine_ovr_accuracy"] == pytest.approx(0.16)
        for row in fixed_pairs
    )


def test_metric_availability_reports_and_rejects_missing_requests() -> None:
    availability = phase9_metric_availability_rows(
        _trajectory_rows(),
        metrics=("after_nine_ovr_accuracy", "after_brier"),
    )
    indexed = {
        (row["cell"], row["metric"]): row for row in availability
    }
    assert indexed[("treatment", "after_nine_ovr_accuracy")][
        "replicas_with_complete_metric"
    ] == 2
    assert indexed[("treatment", "after_brier")]["metric_available"] is False

    require_phase9_metrics(
        availability,
        profile="plan2",
        cells=("control", "treatment"),
        metrics=("after_nine_ovr_accuracy",),
    )
    with pytest.raises(AnalysisArtifactError, match="after_brier"):
        require_phase9_metrics(
            availability,
            profile="plan2",
            cells=("treatment",),
            metrics=("after_brier",),
        )

def test_expected_trajectories_keep_batch_sizes_as_separate_strata() -> None:
    low_data = _trajectory_rows()
    anchor = [
        {
            **row,
            "run_id": f"{row['run_id']}-anchor",
            "control_run_id": (
                None
                if row["control_run_id"] is None
                else f"{row['control_run_id']}-anchor"
            ),
            "samples_per_step": 128,
        }
        for row in low_data
    ]

    expected = phase9_expected_trajectory_rows(
        [*low_data, *anchor],
        metrics=("after_nine_ovr_accuracy",),
    )
    crossings = phase9_durable_crossing_rows(expected, threshold=0.90)

    assert {row["samples_per_step"] for row in expected} == {2, 128}
    assert {row["samples_per_step"] for row in crossings} == {2, 128}
    assert len(crossings) == 4
