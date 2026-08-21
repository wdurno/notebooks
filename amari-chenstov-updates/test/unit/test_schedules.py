import math

import pytest

from src.config import ConfigError, DataConfig, ScheduleConfig
from src.schedules import resolve_schedule, schedule_trajectory_mapping


def _data(schedule: ScheduleConfig) -> DataConfig:
    return DataConfig(
        num_p_steps=100,
        samples_per_step=8,
        non_nine_sampling="empirical",
        initialization_size=100,
        online_pool_size=100,
        reference_pool_size=100,
        evaluation_size=100,
        schedule=schedule,
    )


@pytest.mark.parametrize("steepness", [8.0, 16.0, 32.0])
def test_normalized_logistic_contract(steepness: float) -> None:
    schedule = resolve_schedule(
        _data(
            ScheduleConfig(
                kind="normalized_logistic",
                p_start=0.0,
                p_end=0.2,
                center_fraction=0.5,
                steepness=steepness,
            )
        )
    )

    assert schedule.p_values[0] == 0.0
    assert schedule.p_values[-1] == 0.2
    assert all(
        left < right
        for left, right in zip(schedule.p_values, schedule.p_values[1:])
    )
    assert max(
        abs(schedule.p_values[index] + schedule.p_values[-1 - index] - 0.2)
        for index in range(100)
    ) < 1e-14
    assert math.isclose(sum(schedule.p_values) * 8, 80.0, abs_tol=1e-12)
    assert math.isclose(schedule.center_p, 0.1, abs_tol=1e-14)
    assert schedule.max_speed_transition in {49, 50}


@pytest.mark.parametrize("steepness", [64.0, 128.0, 256.0])
def test_extreme_logistic_schedule_allows_numerical_tail_plateaus(
    steepness: float,
) -> None:
    schedule = resolve_schedule(
        _data(
            ScheduleConfig(
                kind="normalized_logistic",
                p_start=0.0,
                p_end=0.2,
                center_fraction=0.5,
                steepness=steepness,
            )
        )
    )

    assert schedule.p_values[0] == 0.0
    assert schedule.p_values[-1] == 0.2
    assert all(
        left <= right
        for left, right in zip(schedule.p_values, schedule.p_values[1:])
    )
    assert math.isclose(sum(schedule.p_values) * 8, 80.0, abs_tol=1e-12)
    assert schedule.max_speed_transition in {49, 50}


def test_range_matched_linear_schedule_has_equal_exposure() -> None:
    schedule = resolve_schedule(
        _data(
            ScheduleConfig(
                kind="linear",
                p_start=0.0,
                p_end=0.2,
                center_fraction=None,
                steepness=None,
            )
        )
    )

    assert math.isclose(sum(schedule.p_values) * 8, 80.0, abs_tol=1e-12)
    assert schedule.center_p is None
    assert all(
        math.isclose(value, 0.2 / 99.0, abs_tol=1e-15)
        for value in schedule.delta_p_values[1:]
    )


def test_schedule_configuration_rejects_ambiguous_parameters() -> None:
    with pytest.raises(ConfigError, match="null center_fraction"):
        _data(
            ScheduleConfig(
                kind="linear",
                p_start=0.0,
                p_end=0.2,
                center_fraction=0.5,
                steepness=None,
            )
        ).validate()

    with pytest.raises(ConfigError, match="center_fraction"):
        _data(
            ScheduleConfig(
                kind="normalized_logistic",
                p_start=0.0,
                p_end=0.2,
                center_fraction=1.0,
                steepness=16.0,
            )
        ).validate()


def test_schedule_trajectory_records_expected_and_realized_exposure() -> None:
    schedule = resolve_schedule(
        DataConfig(
            num_p_steps=3,
            samples_per_step=2,
            non_nine_sampling="empirical",
            initialization_size=10,
            online_pool_size=10,
            reference_pool_size=10,
            evaluation_size=10,
            schedule=ScheduleConfig(
                kind="linear",
                p_start=0.0,
                p_end=0.2,
                center_fraction=None,
                steepness=None,
            ),
        )
    )
    mapping = schedule_trajectory_mapping(
        schedule,
        ((1, 2), (9, 2), (9, 9)),
        samples_per_step=2,
        stream_plan_hash="a" * 64,
        uniform_stream_hash="b" * 64,
    )

    assert mapping["max_speed_transition"] in {1, 2}
    assert mapping["expected_total_observations_available"] == 6
    assert mapping["expected_total_nines_available"] == pytest.approx(0.6)
    assert mapping["realized_total_nines_available"] == 3
    assert mapping["expected_total_nines_consumed"] == pytest.approx(0.2)
    assert mapping["realized_total_nines_consumed"] == 1
    assert [row["delta_p"] for row in mapping["rows"]] == pytest.approx(
        [0.0, 0.1, 0.1]
    )
