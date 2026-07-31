import pytest
import torch

from src.fisher import LFUBatchEstimate
from src.fixed_trajectory import (
    DENSE_METHODS,
    RIDGE_DENSE_METHODS,
    FixedTrajectory,
    OnlineStepStatistics,
    common_step_metrics,
    lagged_directions,
    replay_dense_conditions,
)


def _trajectory() -> FixedTrajectory:
    parameters = torch.tensor(
        [[0.0, 0.0], [0.1, -0.2], [0.15, -0.1]],
        dtype=torch.float64,
    )
    return FixedTrajectory(
        parameters=parameters,
        displacements=parameters[1:] - parameters[:-1],
        p_values=(0.0, 0.5, 1.0),
        observation_indices=((1, 2), (3, 4), (5, 6)),
        class_labels=((0, 1), (2, 3), (9, 9)),
        driver_steps=(
            {"step": 0},
            {"step": 1},
            {"step": 2},
        ),
        optimizer_state={},
        parameter_layout={"total_numel": 2, "parameters": []},
        stream_plan_hash="stream",
        driver_fisher_cadence=2,
    )


def _statistic(
    fisher: torch.Tensor,
    ac: torch.Tensor,
    residual: torch.Tensor,
) -> OnlineStepStatistics:
    return OnlineStepStatistics(
        estimate=LFUBatchEstimate(
            fisher=fisher,
            amari_chentsov=ac,
            residual=residual,
        ),
        score_gradient_count=4,
        hvp_count=4,
        elapsed_seconds=0.1,
    )


def _replay_fixture():
    initial = torch.diag(torch.tensor([2.0, 1.0], dtype=torch.float64))
    references = [
        initial,
        torch.diag(torch.tensor([2.2, 1.1], dtype=torch.float64)),
        torch.diag(torch.tensor([2.4, 1.2], dtype=torch.float64)),
    ]
    zeros = torch.zeros(2, 2, dtype=torch.float64)
    statistics = [
        _statistic(initial, zeros, zeros),
        _statistic(
            torch.diag(torch.tensor([2.1, 1.05], dtype=torch.float64)),
            torch.diag(torch.tensor([0.1, 0.05], dtype=torch.float64)),
            torch.diag(torch.tensor([0.04, 0.02], dtype=torch.float64)),
        ),
        _statistic(
            torch.diag(torch.tensor([2.3, 1.15], dtype=torch.float64)),
            torch.diag(torch.tensor([0.08, 0.04], dtype=torch.float64)),
            torch.diag(torch.tensor([0.02, 0.01], dtype=torch.float64)),
        ),
    ]
    trajectory = _trajectory()
    directions = lagged_directions(
        trajectory,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    results = replay_dense_conditions(
        initial,
        references,
        statistics,
        directions,
        trajectory.p_values,
        ema_gain=0.25,
        fresh_fisher_cadence=2,
    )
    return initial, references, statistics, trajectory, directions, results


def test_lagged_directions_use_only_the_accepted_preceding_move() -> None:
    trajectory = _trajectory()

    directions = lagged_directions(
        trajectory,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    torch.testing.assert_close(directions[0], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(directions[1], trajectory.displacements[0])
    torch.testing.assert_close(directions[2], trajectory.displacements[1])


def test_trajectory_hash_and_validation_detect_parameter_mutation() -> None:
    trajectory = _trajectory()
    first_hash = trajectory.content_hash
    same = _trajectory()

    assert same.content_hash == first_hash
    broken_parameters = trajectory.parameters.clone()
    broken_parameters[1, 0] += 1
    broken = FixedTrajectory(
        **{
            **trajectory.__dict__,
            "parameters": broken_parameters,
        }
    )
    with pytest.raises(ValueError, match="displacements"):
        broken.validate()


def test_dense_replay_applies_each_treatment_once_and_preserves_t0() -> None:
    initial, references, statistics, trajectory, directions, results = (
        _replay_fixture()
    )

    assert tuple(results) == DENSE_METHODS
    for result in results.values():
        torch.testing.assert_close(result.estimates[0], initial)
    torch.testing.assert_close(trajectory.parameters, _trajectory().parameters)
    expected_reference = torch.diag(
        torch.tensor([2.2, 1.1], dtype=torch.float64)
    )
    torch.testing.assert_close(references[1], expected_reference)
    torch.testing.assert_close(directions[1], trajectory.displacements[0])

    ema_prediction = initial
    ac_prediction = initial + statistics[1].estimate.amari_chentsov
    full_prediction = ac_prediction + statistics[1].estimate.residual
    torch.testing.assert_close(results["ema"].predictions[1], ema_prediction)
    torch.testing.assert_close(
        results["periodic_fresh"].predictions[1],
        ema_prediction,
    )
    torch.testing.assert_close(results["ac_only"].predictions[1], ac_prediction)
    torch.testing.assert_close(results["full_lfu"].predictions[1], full_prediction)


def test_periodic_fresh_uses_ema_between_exact_replacements() -> None:
    _, references, _, _, _, results = _replay_fixture()

    assert results["periodic_fresh"].metrics[1]["fresh_replacement"] is False
    assert results["periodic_fresh"].metrics[2]["fresh_replacement"] is True
    torch.testing.assert_close(
        results["periodic_fresh"].estimates[2],
        references[2],
    )


def test_dense_replay_adds_ridge_conditions_without_changing_raw_controls() -> None:
    initial, references, statistics, trajectory, directions, raw = (
        _replay_fixture()
    )

    results = replay_dense_conditions(
        initial,
        references,
        statistics,
        directions,
        trajectory.p_values,
        ema_gain=0.25,
        fresh_fisher_cadence=2,
        ridge_half_life_steps=1.0,
        ridge_amplitude_epsilon=1e-9,
        ridge_coherence_threshold=0.75,
    )

    assert tuple(results) == DENSE_METHODS + RIDGE_DENSE_METHODS
    for method in DENSE_METHODS:
        for actual, expected in zip(
            results[method].estimates,
            raw[method].estimates,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected)

    first = results["ridge_full_lfu"].metrics[1]["ridge"]
    assert first["cold_started"]
    assert first["warmup_mass"] == pytest.approx(0.5)
    assert results["ridge_ac_only"].ridge_states is None
    assert results["ridge_full_lfu"].ridge_states[0] is None
    assert results["ridge_full_lfu"].ridge_states[1] is not None

    expected_ac = 0.5 * statistics[1].estimate.amari_chentsov
    expected_full = 0.5 * statistics[1].estimate.full
    torch.testing.assert_close(
        results["ridge_ac_only"].predictions[1],
        initial + expected_ac,
    )
    torch.testing.assert_close(
        results["ridge_full_lfu"].predictions[1],
        initial + expected_full,
    )


def test_dense_replay_rejects_partial_ridge_settings() -> None:
    initial, references, statistics, trajectory, directions, _ = (
        _replay_fixture()
    )

    with pytest.raises(ValueError, match="all omitted or all supplied"):
        replay_dense_conditions(
            initial,
            references,
            statistics,
            directions,
            trajectory.p_values,
            ema_gain=0.25,
            fresh_fisher_cadence=2,
            ridge_half_life_steps=2.0,
        )


def test_common_metrics_record_ac_and_residual_independently() -> None:
    _, references, statistics, trajectory, directions, _ = _replay_fixture()

    rows = common_step_metrics(
        references,
        statistics,
        directions,
        trajectory.p_values,
    )

    assert rows[0]["full_increment_relative_error"] is None
    assert rows[1]["ac_fro"] > 0
    assert rows[1]["residual_fro"] > 0
    assert rows[1]["full_lfu_fro"] > rows[1]["ac_fro"]
