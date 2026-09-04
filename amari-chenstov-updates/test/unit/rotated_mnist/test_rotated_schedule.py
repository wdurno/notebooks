import dataclasses
from pathlib import Path

import pytest

from mnist_experiment.rotated_mnist.config import RotationConfig
from mnist_experiment.rotated_mnist.config import load_config
from mnist_experiment.rotated_mnist.schedule import (
    PRINCIPAL_KNOTS_DEGREES,
    RotationSchedule,
    resolve_rotation_schedule,
    resolve_shaped_rotation_schedule,
)


SMOKE_CONFIG = (
    Path(__file__).parents[3]
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase1_smoke.json"
)


def test_principal_schedule_has_exact_knots_and_speed_contract() -> None:
    config = load_config(SMOKE_CONFIG)
    rotation = dataclasses.replace(
        config.rotation,
        transitions_per_arrow=20,
    )
    schedule = resolve_rotation_schedule(rotation)

    assert schedule.num_points == 101
    assert schedule.num_transitions == 100
    assert tuple(schedule.angles_degrees[index] for index in range(0, 101, 20)) == (
        PRINCIPAL_KNOTS_DEGREES
    )
    assert set(
        abs(right - left)
        for left, right in zip(
            schedule.angles_degrees[:-1],
            schedule.angles_degrees[1:],
            strict=True,
        )
    ) == {0.75, 1.5}
    assert schedule.directions_to_next[19] == 1
    assert schedule.directions_to_next[40] == -1
    assert schedule.directions_to_next[60] == 1
    assert schedule.directions_to_next[-1] == 0


def test_smoke_schedule_exercises_both_changes_and_round_trips() -> None:
    schedule = resolve_rotation_schedule(load_config(SMOKE_CONFIG).rotation)

    assert schedule.angles_degrees == PRINCIPAL_KNOTS_DEGREES
    assert schedule.directions_to_next == (1, 1, -1, 1, 1, 0)
    assert RotationSchedule.from_mapping(schedule.to_mapping()) == schedule
    assert RotationSchedule.from_mapping(schedule.to_mapping()).content_hash == (
        schedule.content_hash
    )


def test_sigmoid_schedule_preserves_knots_and_total_distance() -> None:
    config = RotationConfig(
        knots_degrees=(0.0, 30.0, 0.0, 30.0),
        transitions_per_arrow=40,
        interpolation="bilinear",
        expand=False,
        fill=0.0,
    )
    linear = resolve_shaped_rotation_schedule(
        config, kind="linear", sigmoid_kappa=8.0
    )
    sigmoid = resolve_shaped_rotation_schedule(
        config, kind="sigmoid", sigmoid_kappa=8.0
    )

    assert sigmoid.num_transitions == 120
    assert tuple(sigmoid.angles_degrees[index] for index in (0, 40, 80, 120)) == (
        0.0,
        30.0,
        0.0,
        30.0,
    )
    assert sigmoid.cumulative_degrees[-1] == pytest.approx(90.0)
    assert linear.cumulative_degrees[-1] == pytest.approx(90.0)
    assert sigmoid.angles_degrees[20] == pytest.approx(15.0)
    assert sigmoid.angles_degrees[10] < linear.angles_degrees[10]
    assert sigmoid.angles_degrees[30] > linear.angles_degrees[30]
