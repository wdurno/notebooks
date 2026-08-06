"""Predictable fixed-batch adaptation controller for unified Phase 8 runs."""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor

from .config import ControllerConfig


def half_life_gain(delta_p: float, half_life_p: float) -> float:
    """Return an EMA gain parameterized by environmental distance."""

    if not math.isfinite(delta_p) or delta_p < 0.0:
        raise ValueError("delta_p must be finite and nonnegative")
    if not math.isfinite(half_life_p) or half_life_p <= 0.0:
        raise ValueError("half_life_p must be finite and positive")
    return 1.0 - 2.0 ** (-delta_p / half_life_p)


def effective_size_update(q: float, pi: float, batch_size: int) -> float:
    """Advance the scalar covariance-mass recursion."""

    if not math.isfinite(q) or q <= 0.0:
        raise ValueError("q must be finite and positive")
    if not math.isfinite(pi) or not 0.0 <= pi <= 1.0:
        raise ValueError("pi must be in [0, 1]")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return (1.0 - pi) ** 2 * q + pi**2 / batch_size


def fixed_batch_optimal_pi(
    signal_squared: float,
    trace_inverse_fisher: float,
    q: float,
    batch_size: int,
    *,
    epsilon: float,
) -> float:
    """Minimize the applied one-step fixed-batch Euclidean risk."""

    values = (signal_squared, trace_inverse_fisher, q, epsilon)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("controller risk inputs must be finite")
    if signal_squared < 0.0 or trace_inverse_fisher < 0.0:
        raise ValueError("signal and trace inputs must be nonnegative")
    if q <= 0.0 or epsilon <= 0.0:
        raise ValueError("q and epsilon must be positive")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    old_risk = signal_squared + trace_inverse_fisher * q
    new_risk = trace_inverse_fisher / batch_size
    return old_risk / (old_risk + new_risk + epsilon)


def bernoulli_theory_optimal_pi(
    signal_squared: float,
    trace_inverse_fisher: float,
    effective_size: float,
) -> float:
    """Return the fixed-total Bernoulli oracle used only as a diagnostic."""

    values = (signal_squared, trace_inverse_fisher, effective_size)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("theory-oracle inputs must be finite")
    if signal_squared < 0.0 or trace_inverse_fisher < 0.0:
        raise ValueError("signal and trace inputs must be nonnegative")
    if effective_size <= 0.0:
        raise ValueError("effective_size must be positive")
    if signal_squared == 0.0:
        return 0.0
    raw = 1.0 - trace_inverse_fisher / (
        2.0 * effective_size * signal_squared
    )
    return min(1.0, max(0.0, raw))


@dataclasses.dataclass(frozen=True)
class OracleControllerInput:
    displacement: Tensor

    def validate(self, parameter_count: int) -> None:
        if self.displacement.shape != (parameter_count,):
            raise ValueError("oracle displacement has an invalid shape")
        if not self.displacement.is_floating_point() or not torch.isfinite(
            self.displacement
        ).all():
            raise ValueError("oracle displacement must be finite and floating")


@dataclasses.dataclass(frozen=True)
class ControllerState:
    trend: Tensor
    q: float
    residual_moment: float
    scale_moment: float
    environment_distance: float
    previous_pi: float | None
    accepted_steps: int
    oracle_residual_moment: float = 0.0
    oracle_scale_moment: float = 0.0

    @classmethod
    def initialize(
        cls,
        parameter_count: int,
        initial_effective_size: float,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> "ControllerState":
        if (
            not isinstance(parameter_count, int)
            or isinstance(parameter_count, bool)
            or parameter_count < 1
        ):
            raise ValueError("parameter_count must be a positive integer")
        if not math.isfinite(initial_effective_size) or initial_effective_size <= 0:
            raise ValueError("initial_effective_size must be finite and positive")
        return cls(
            trend=torch.zeros(parameter_count, dtype=dtype, device=device),
            q=1.0 / float(initial_effective_size),
            residual_moment=0.0,
            scale_moment=0.0,
            environment_distance=0.0,
            previous_pi=None,
            accepted_steps=0,
            oracle_residual_moment=0.0,
            oracle_scale_moment=0.0,
        )

    @property
    def effective_size(self) -> float:
        return 1.0 / self.q

    def trace_estimate(self, epsilon: float) -> float:
        return self.residual_moment / (self.scale_moment + epsilon)

    def oracle_trace_estimate(self, epsilon: float) -> float | None:
        if self.oracle_scale_moment == 0.0:
            return None
        return self.oracle_residual_moment / (
            self.oracle_scale_moment + epsilon
        )

    def scalar_mapping(self, epsilon: float) -> dict[str, float | int | None]:
        return {
            "q": self.q,
            "effective_size": self.effective_size,
            "residual_moment": self.residual_moment,
            "scale_moment": self.scale_moment,
            "trace_estimate": self.trace_estimate(epsilon),
            "oracle_residual_moment": self.oracle_residual_moment,
            "oracle_scale_moment": self.oracle_scale_moment,
            "oracle_trace_estimate": self.oracle_trace_estimate(epsilon),
            "environment_distance": self.environment_distance,
            "previous_pi": self.previous_pi,
            "accepted_steps": self.accepted_steps,
            "trend_norm": float(torch.linalg.vector_norm(self.trend)),
        }


@dataclasses.dataclass(frozen=True)
class ControllerDecision:
    policy: str
    raw_pi: float
    applied_pi: float
    plugin_pi: float
    cold_start_pi: float
    oracle_pi: float | None
    theory_oracle_pi: float | None
    lower_bound_active: bool
    upper_bound_active: bool
    cold_start_active: bool
    ewc_odds: float | None
    signal_squared: float
    trace_estimate: float
    oracle_trace_estimate: float | None
    old_covariance_trace: float
    new_covariance_trace: float
    effective_size: float

    def mapping(self) -> dict[str, float | str | bool | None]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ControllerAcceptance:
    state: ControllerState
    gain: float
    residual: Tensor
    residual_squared: float
    scale_observation: float
    normalized_displacement: Tensor | None
    oracle_residual: Tensor | None


def _bounded(value: float, lower: float, upper: float) -> tuple[float, bool, bool]:
    applied = min(upper, max(lower, value))
    return applied, value < lower, value > upper


def decide_controller(
    state: ControllerState,
    config: ControllerConfig,
    *,
    batch_size: int,
    oracle: OracleControllerInput | None = None,
) -> ControllerDecision:
    """Choose pi using only state produced by previously accepted steps."""

    config.validate()
    if config.pi_min is None or config.trend_half_life_p is None:
        raise ValueError("unified controller settings are required")
    if config.trace_epsilon is None:
        raise ValueError("trace_epsilon is required")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if oracle is not None:
        oracle.validate(state.trend.numel())

    trace = state.trace_estimate(config.trace_epsilon)
    signal_squared = float(state.trend @ state.trend)
    plugin_pi = fixed_batch_optimal_pi(
        signal_squared,
        trace,
        state.q,
        batch_size,
        epsilon=config.trace_epsilon,
    )
    cold_pi = batch_size / (state.effective_size + batch_size)
    cold_start = state.environment_distance < config.trend_half_life_p

    oracle_pi = None
    theory_pi = None
    oracle_trace = state.oracle_trace_estimate(config.trace_epsilon)
    if oracle is not None:
        oracle_signal_squared = float(oracle.displacement @ oracle.displacement)
        if oracle_trace is not None:
            oracle_pi = fixed_batch_optimal_pi(
                oracle_signal_squared,
                oracle_trace,
                state.q,
                batch_size,
                epsilon=config.trace_epsilon,
            )
            theory_pi = bernoulli_theory_optimal_pi(
                oracle_signal_squared,
                oracle_trace,
                state.effective_size,
            )

    if config.policy == "uncontrolled":
        raw_pi = applied_pi = 1.0
        lower_active = upper_active = False
    elif config.policy == "freeze":
        raw_pi = applied_pi = 0.0
        lower_active = upper_active = False
    else:
        if config.policy == "fixed_unified":
            raw_pi = float(config.fixed_pi)
        elif config.policy == "optimal_oracle":
            if not cold_start and oracle_pi is None:
                raise ValueError("optimal_oracle requires oracle inputs")
            raw_pi = cold_pi if cold_start else float(oracle_pi)
        elif config.policy == "optimal_plugin":
            raw_pi = cold_pi if cold_start else plugin_pi
        else:
            raise ValueError(f"policy is not unified: {config.policy}")
        applied_pi, lower_active, upper_active = _bounded(
            raw_pi,
            float(config.pi_min),
            float(config.pi_max),
        )

    ewc_odds = (
        None
        if applied_pi == 0.0
        else (1.0 - applied_pi) / applied_pi
    )
    return ControllerDecision(
        policy=config.policy,
        raw_pi=raw_pi,
        applied_pi=applied_pi,
        plugin_pi=plugin_pi,
        cold_start_pi=cold_pi,
        oracle_pi=oracle_pi,
        theory_oracle_pi=theory_pi,
        lower_bound_active=lower_active,
        upper_bound_active=upper_active,
        cold_start_active=cold_start,
        ewc_odds=ewc_odds,
        signal_squared=signal_squared,
        trace_estimate=trace,
        oracle_trace_estimate=oracle_trace,
        old_covariance_trace=trace * state.q,
        new_covariance_trace=trace / batch_size,
        effective_size=state.effective_size,
    )


def accept_controller_step(
    state: ControllerState,
    decision: ControllerDecision,
    displacement: Tensor,
    *,
    batch_size: int,
    delta_p: float,
    half_life_p: float,
    oracle_displacement: Tensor | None = None,
) -> ControllerAcceptance:
    """Update trend, trace moments, and effective size after accepting a step."""

    if displacement.shape != state.trend.shape:
        raise ValueError("displacement has an invalid shape")
    if displacement.dtype != state.trend.dtype or displacement.device != state.trend.device:
        raise ValueError("displacement must share trend dtype and device")
    if not torch.isfinite(displacement).all():
        raise ValueError("displacement must be finite")
    if oracle_displacement is not None and (
        oracle_displacement.shape != state.trend.shape
        or oracle_displacement.dtype != state.trend.dtype
        or oracle_displacement.device != state.trend.device
        or not torch.isfinite(oracle_displacement).all()
    ):
        raise ValueError(
            "oracle_displacement must be finite and share trend shape/dtype/device"
        )
    gain = half_life_gain(delta_p, half_life_p)
    pi = decision.applied_pi
    if pi == 0.0:
        residual = torch.zeros_like(displacement)
        residual_squared = 0.0
        scale = 0.0
        normalized_displacement = None
        oracle_residual = None
        residual_moment = state.residual_moment
        scale_moment = state.scale_moment
        oracle_residual_moment = state.oracle_residual_moment
        oracle_scale_moment = state.oracle_scale_moment
        trend = state.trend
    else:
        # Locally, the EWC minimizer moves by pi times the unregularized drift.
        normalized_displacement = displacement / pi
        residual = displacement - pi * state.trend
        residual_squared = float(residual @ residual)
        scale = pi**2 * (state.q + 1.0 / batch_size)
        residual_moment = (
            (1.0 - gain) * state.residual_moment + gain * residual_squared
        )
        scale_moment = (1.0 - gain) * state.scale_moment + gain * scale
        if oracle_displacement is None:
            oracle_residual = None
            oracle_residual_moment = state.oracle_residual_moment
            oracle_scale_moment = state.oracle_scale_moment
        else:
            oracle_residual = displacement - pi * oracle_displacement
            oracle_residual_squared = float(oracle_residual @ oracle_residual)
            oracle_residual_moment = (
                (1.0 - gain) * state.oracle_residual_moment
                + gain * oracle_residual_squared
            )
            oracle_scale_moment = (
                (1.0 - gain) * state.oracle_scale_moment + gain * scale
            )
        trend = (
            (1.0 - gain) * state.trend
            + gain * normalized_displacement
        )
    q = effective_size_update(state.q, decision.applied_pi, batch_size)
    next_state = ControllerState(
        trend=trend,
        q=q,
        residual_moment=residual_moment,
        scale_moment=scale_moment,
        environment_distance=state.environment_distance + delta_p,
        previous_pi=decision.applied_pi,
        accepted_steps=state.accepted_steps + 1,
        oracle_residual_moment=oracle_residual_moment,
        oracle_scale_moment=oracle_scale_moment,
    )
    return ControllerAcceptance(
        state=next_state,
        gain=gain,
        residual=residual,
        residual_squared=residual_squared,
        scale_observation=scale,
        normalized_displacement=normalized_displacement,
        oracle_residual=oracle_residual,
    )
