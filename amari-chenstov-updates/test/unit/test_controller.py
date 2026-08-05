import dataclasses
import math

import pytest
import torch

from src.config import ControllerConfig
from src.controller import (
    ControllerState,
    OracleControllerInput,
    accept_controller_step,
    bernoulli_theory_optimal_pi,
    decide_controller,
    effective_size_update,
    fixed_batch_optimal_pi,
    half_life_gain,
)


def _config(policy: str = "optimal_plugin", **overrides: object) -> ControllerConfig:
    values = {
        "policy": policy,
        "fixed_pi": 0.5,
        "pi_min": 0.05,
        "pi_max": 0.95,
        "trend_half_life_p": 0.2,
        "trace_epsilon": 1e-12,
        "oracle_mode": "reference_path" if policy == "optimal_oracle" else "none",
        "damping": None,
        "epsilon": None,
    }
    values.update(overrides)
    return ControllerConfig(**values)


def test_fixed_batch_formula_minimizes_quadratic_risk() -> None:
    signal_squared = 0.3
    trace = 2.0
    q = 0.1
    batch_size = 5
    pi = fixed_batch_optimal_pi(
        signal_squared,
        trace,
        q,
        batch_size,
        epsilon=1e-15,
    )
    expected = (signal_squared + trace * q) / (
        signal_squared + trace * q + trace / batch_size
    )
    assert pi == pytest.approx(expected)

    def risk(value: float) -> float:
        return (1.0 - value) ** 2 * (signal_squared + trace * q) + (
            value**2 * trace / batch_size
        )

    assert risk(pi) < risk(pi - 0.05)
    assert risk(pi) < risk(pi + 0.05)


def test_half_life_and_effective_size_recursions_match_hand_values() -> None:
    assert half_life_gain(0.2, 0.2) == pytest.approx(0.5)
    assert effective_size_update(0.1, 0.25, 4) == pytest.approx(
        0.75**2 * 0.1 + 0.25**2 / 4
    )


def test_cold_start_is_bounded_and_explicit_boundaries_bypass_clipping() -> None:
    state = ControllerState.initialize(2, initial_effective_size=100.0)
    plugin = decide_controller(state, _config(), batch_size=10)
    assert plugin.raw_pi == pytest.approx(10.0 / 110.0)
    assert plugin.applied_pi == pytest.approx(10.0 / 110.0)
    assert plugin.cold_start_active

    uncontrolled = decide_controller(
        state,
        _config("uncontrolled"),
        batch_size=10,
    )
    frozen = decide_controller(state, _config("freeze"), batch_size=10)
    assert uncontrolled.applied_pi == 1.0
    assert frozen.applied_pi == 0.0
    assert frozen.ewc_odds is None


def test_acceptance_uses_predictable_trend_and_updates_all_moments() -> None:
    state = ControllerState.initialize(2, initial_effective_size=10.0)
    decision = decide_controller(
        state,
        _config("fixed_unified", fixed_pi=0.25),
        batch_size=4,
    )
    displacement = torch.tensor([1.0, -1.0], dtype=torch.float64)
    accepted = accept_controller_step(
        state,
        decision,
        displacement,
        batch_size=4,
        delta_p=0.2,
        half_life_p=0.2,
    )

    scale = 0.25**2 * (0.1 + 0.25)
    assert accepted.gain == pytest.approx(0.5)
    assert accepted.residual_squared == pytest.approx(2.0)
    assert accepted.scale_observation == pytest.approx(scale)
    torch.testing.assert_close(
        accepted.state.trend,
        torch.tensor([0.5, -0.5], dtype=torch.float64),
    )
    assert accepted.state.residual_moment == pytest.approx(1.0)
    assert accepted.state.scale_moment == pytest.approx(0.5 * scale)
    assert accepted.state.q == pytest.approx(
        effective_size_update(0.1, 0.25, 4)
    )


def test_decision_does_not_mutate_state_or_read_current_displacement() -> None:
    state = ControllerState(
        trend=torch.tensor([0.2, 0.0], dtype=torch.float64),
        q=0.1,
        residual_moment=0.4,
        scale_moment=0.2,
        environment_distance=0.2,
        previous_pi=0.5,
        accepted_steps=2,
    )
    before = state.trend.clone()
    first = decide_controller(state, _config(), batch_size=4)
    second = decide_controller(state, _config(), batch_size=4)
    assert first == second
    torch.testing.assert_close(state.trend, before)


def test_oracle_and_theory_diagnostics_use_reference_inputs() -> None:
    state = dataclasses.replace(
        ControllerState.initialize(2, initial_effective_size=10.0),
        environment_distance=0.3,
        oracle_residual_moment=2.0,
        oracle_scale_moment=1.0,
    )
    oracle = OracleControllerInput(
        displacement=torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    decision = decide_controller(
        state,
        _config("optimal_oracle"),
        batch_size=5,
        oracle=oracle,
    )
    expected = fixed_batch_optimal_pi(1.0, 2.0, 0.1, 5, epsilon=1e-12)
    assert decision.raw_pi == pytest.approx(expected)
    assert decision.theory_oracle_pi == pytest.approx(
        bernoulli_theory_optimal_pi(1.0, 2.0, 10.0)
    )


def test_variable_pi_calibrated_process_recovers_trend_and_trace() -> None:
    generator = torch.Generator().manual_seed(8128)
    true_trend = torch.tensor([0.03, -0.02, 0.01], dtype=torch.float64)
    true_trace = 4.0
    state = dataclasses.replace(
        ControllerState.initialize(3, initial_effective_size=50.0),
        trend=true_trend.clone(),
        environment_distance=1.0,
    )
    pi_values = (0.1, 0.35, 0.7, 0.2)
    for step in range(4000):
        pi = pi_values[step % len(pi_values)]
        decision = decide_controller(
            state,
            _config("fixed_unified", fixed_pi=pi),
            batch_size=8,
        )
        scale = pi**2 * (state.q + 1.0 / 8.0)
        noise = torch.randn(3, generator=generator, dtype=torch.float64)
        noise *= math.sqrt(scale * true_trace / 3.0)
        accepted = accept_controller_step(
            state,
            decision,
            true_trend + noise,
            batch_size=8,
            delta_p=0.002,
            half_life_p=0.2,
            oracle_displacement=true_trend,
        )
        state = accepted.state

    # The predictable EMA trend contributes finite-window residual energy. The
    # assumption checks measure this gap rather than pretending it is absent.
    assert state.trace_estimate(1e-12) == pytest.approx(true_trace, rel=0.20)
    assert state.oracle_trace_estimate(1e-12) == pytest.approx(
        true_trace,
        rel=0.20,
    )
    assert abs(state.oracle_trace_estimate(1e-12) - true_trace) <= abs(
        state.trace_estimate(1e-12) - true_trace
    )
    torch.testing.assert_close(state.trend, true_trend, rtol=0.0, atol=0.08)


def test_near_pi_min_state_remains_finite_and_retains_prior_information() -> None:
    state = ControllerState(
        trend=torch.zeros(2, dtype=torch.float64),
        q=1e-4,
        residual_moment=2.0,
        scale_moment=1.0,
        environment_distance=1.0,
        previous_pi=0.05,
        accepted_steps=20,
    )
    decision = decide_controller(state, _config(), batch_size=1)
    assert decision.applied_pi == 0.05
    accepted = accept_controller_step(
        state,
        decision,
        torch.zeros(2, dtype=torch.float64),
        batch_size=1,
        delta_p=0.01,
        half_life_p=0.2,
    )
    assert math.isfinite(accepted.state.trace_estimate(1e-12))
    assert accepted.state.q > 0.0
    assert accepted.state.residual_moment > 0.0


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"pi_min": 0.8, "pi_max": 0.2}, "cannot exceed"),
        ({"trend_half_life_p": 0.0}, "half_life"),
        ({"trace_epsilon": 0.0}, "trace_epsilon"),
    ],
)
def test_invalid_unified_settings_fail(overrides: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        decide_controller(
            ControllerState.initialize(1, initial_effective_size=2.0),
            _config(**overrides),
            batch_size=1,
        )
