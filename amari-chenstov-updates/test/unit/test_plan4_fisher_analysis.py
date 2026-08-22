import math

import torch

from src.controller import ControllerState
from src.plan4_fisher_analysis import (
    HALF_LIVES,
    PRIMARY_HALF_LIFE,
    SCHEDULE_NAMES,
    SENSITIVITY_HALF_LIFE,
    _fixed_policy_risk,
    _interpolate_parameter_path,
    expected_controller_update,
    phase2_schedules,
)
from src.representations import DiagonalFisher


def test_plan4_fisher_parameter_path_interpolation_is_piecewise_linear() -> None:
    source_p = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
    parameters = torch.tensor(
        [[0.0, 1.0], [2.0, 3.0], [6.0, 7.0]], dtype=torch.float64
    )
    target_p = torch.tensor([0.0, 0.05, 0.15, 0.2], dtype=torch.float64)

    result = _interpolate_parameter_path(source_p, parameters, target_p)

    assert torch.allclose(
        result,
        torch.tensor(
            [[0.0, 1.0], [1.0, 2.0], [4.0, 5.0], [6.0, 7.0]],
            dtype=torch.float64,
        ),
    )


def test_expected_fisher_controller_moments_use_one_predictable_metric() -> None:
    state = ControllerState.initialize(2, 10.0, dtype=torch.float64)
    fisher = DiagonalFisher(torch.tensor([1.0, 0.0], dtype=torch.float64))

    updated, diagnostics = expected_controller_update(
        state,
        0.25,
        torch.tensor([2.0, 3.0], dtype=torch.float64),
        fisher,
        2.0,
        batch_size=4,
        delta_p=0.2,
        half_life_p=0.2,
    )

    assert torch.equal(updated.trend, torch.tensor([1.0, 1.5], dtype=torch.float64))
    assert math.isclose(updated.q, 0.071875)
    assert math.isclose(updated.residual_moment, 0.146875)
    assert math.isclose(updated.scale_moment, 0.0109375)
    assert math.isclose(updated.oracle_residual_moment, 0.021875)
    assert math.isclose(diagnostics["trend_error_energy"], 4.0)
    assert math.isclose(diagnostics["expected_residual_energy"], 0.29375)


def test_plan4_fisher_screen_freezes_scaled_half_lives_and_stress_schedules() -> None:
    schedules = phase2_schedules()

    assert HALF_LIVES == (PRIMARY_HALF_LIFE, SENSITIVITY_HALF_LIFE) == (0.05, 0.10)
    assert tuple(SCHEDULE_NAMES) == (
        "linear",
        "logistic-k32",
        "logistic-k64",
        "logistic-k128",
        "logistic-k256",
    )
    assert all(len(schedules[name].p_values) == 100 for name in SCHEDULE_NAMES)
    assert all(schedules[name].p_values[-1] == 0.2 for name in SCHEDULE_NAMES)


def test_fixed_policy_risk_rolls_its_own_covariance_state() -> None:
    signal = torch.tensor([0.2, 0.4], dtype=torch.float64).numpy()
    dimension = torch.tensor([3.0, 5.0], dtype=torch.float64).numpy()
    event = torch.tensor([True, False]).numpy()

    event_risk, full_risk = _fixed_policy_risk(
        signal, dimension, event, 0.05
    )

    initial_q = 1.0 / 30_000
    first = 0.95**2 * (0.2 + 3.0 * initial_q) + 0.05**2 * 3.0 / 8
    next_q = 0.95**2 * initial_q + 0.05**2 / 8
    second = 0.95**2 * (0.4 + 5.0 * next_q) + 0.05**2 * 5.0 / 8
    assert math.isclose(event_risk, first)
    assert math.isclose(full_risk, first + second)
