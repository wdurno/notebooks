"""Predictable fixed-batch adaptation controller for unified Phase 8 runs."""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor

from .config import ControllerConfig
from .representations import FisherRepresentation


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


def update_step_half_life_gain(half_life_steps: float) -> float:
    """Return the constant EMA gain for a half-life in accepted updates."""

    if not math.isfinite(half_life_steps) or half_life_steps <= 0.0:
        raise ValueError("half_life_steps must be finite and positive")
    return 1.0 - 2.0 ** (-1.0 / half_life_steps)


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

    def scalar_mapping(
        self,
        epsilon: float,
        *,
        risk_metric: str | None = None,
    ) -> dict[str, float | int | str | None]:
        mapping: dict[str, float | int | str | None] = {
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
        if risk_metric is not None:
            mapping.update(
                {
                    "risk_metric": risk_metric,
                    "uncertainty_scale_estimate": self.trace_estimate(epsilon),
                    "uncertainty_scale_semantics": (
                        "trace_inverse_fisher"
                        if risk_metric == "euclidean"
                        else "fisher_weighted_covariance_dimension"
                    ),
                }
            )
        return mapping


@dataclasses.dataclass(frozen=True)
class DiscountedRiskState:
    old_risk_moment: float = 0.0
    new_risk_moment: float = 0.0
    updates: int = 0

    def validate(self) -> None:
        if (
            not math.isfinite(self.old_risk_moment)
            or not math.isfinite(self.new_risk_moment)
            or self.old_risk_moment < 0.0
            or self.new_risk_moment < 0.0
        ):
            raise ValueError("discounted risk moments must be finite and nonnegative")
        if (
            not isinstance(self.updates, int)
            or isinstance(self.updates, bool)
            or self.updates < 0
        ):
            raise ValueError("discounted risk updates must be a nonnegative integer")

    def mapping(self) -> dict[str, float | int]:
        self.validate()
        denominator = self.old_risk_moment + self.new_risk_moment
        return {
            "old_risk_moment": self.old_risk_moment,
            "new_risk_moment": self.new_risk_moment,
            "risk_moment_denominator": denominator,
            "unclipped_pi": (
                self.old_risk_moment / denominator
                if denominator > 0.0
                else 0.0
            ),
            "updates": self.updates,
        }


@dataclasses.dataclass(frozen=True)
class DiscountedRiskDecision:
    controller: "ControllerDecision"
    state: DiscountedRiskState
    gain: float
    instantaneous_old_risk: float
    instantaneous_new_risk: float
    unclipped_pi: float | None
    zero_denominator_fallback: bool

    def mapping(self) -> dict[str, float | int | bool | None]:
        return {
            "edr_gain": self.gain,
            "edr_instantaneous_old_risk": self.instantaneous_old_risk,
            "edr_instantaneous_new_risk": self.instantaneous_new_risk,
            "edr_old_risk_moment": self.state.old_risk_moment,
            "edr_new_risk_moment": self.state.new_risk_moment,
            "edr_updates": self.state.updates,
            "edr_unclipped_pi": self.unclipped_pi,
            "edr_zero_denominator_fallback": self.zero_denominator_fallback,
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
    risk_metric: str
    uncertainty_scale_semantics: str
    zero_information_fallback: bool

    def mapping(
        self, *, extended: bool = False
    ) -> dict[str, float | str | bool | None]:
        mapping = dataclasses.asdict(self)
        if extended:
            mapping.update(
                {
                    "signal_energy": self.signal_squared,
                    "uncertainty_scale_estimate": self.trace_estimate,
                    "old_covariance_risk": self.old_covariance_trace,
                    "new_covariance_risk": self.new_covariance_trace,
                    "risk_numerator": (
                        self.signal_squared + self.old_covariance_trace
                    ),
                    "risk_denominator": (
                        self.signal_squared
                        + self.old_covariance_trace
                        + self.new_covariance_trace
                    ),
                }
            )
        else:
            for name in (
                "risk_metric",
                "uncertainty_scale_semantics",
                "zero_information_fallback",
            ):
                mapping.pop(name)
        return mapping


@dataclasses.dataclass(frozen=True)
class ControllerAcceptance:
    state: ControllerState
    gain: float
    residual: Tensor
    residual_squared: float
    scale_observation: float
    normalized_displacement: Tensor | None
    oracle_residual: Tensor | None
    residual_euclidean_squared: float
    oracle_residual_energy: float | None
    oracle_residual_euclidean_squared: float | None


def _bounded(value: float, lower: float, upper: float) -> tuple[float, bool, bool]:
    applied = min(upper, max(lower, value))
    return applied, value < lower, value > upper


def _quadratic_energy(
    vector: Tensor,
    risk_metric: str,
    fisher: FisherRepresentation | None,
) -> float:
    if risk_metric == "euclidean":
        return float(vector @ vector)
    if risk_metric != "fisher":
        raise ValueError(f"unsupported controller risk metric: {risk_metric}")
    if fisher is None:
        raise ValueError("fisher-risk controller requires a predictable Fisher")
    if fisher.shape != (vector.numel(), vector.numel()):
        raise ValueError("predictable Fisher has an invalid shape")
    metric_vector = vector.to(device=fisher.device, dtype=fisher.dtype)
    value = float(fisher.quadratic(metric_vector))
    if not math.isfinite(value):
        raise ValueError("Fisher quadratic must be finite")
    scale = max(
        float(fisher.diagonal_vector().abs().max())
        * float(metric_vector.square().sum()),
        1.0,
    )
    tolerance = 100.0 * torch.finfo(fisher.dtype).eps * scale
    if value < -tolerance:
        raise ValueError("Fisher quadratic is materially negative")
    return max(value, 0.0)


def decide_controller(
    state: ControllerState,
    config: ControllerConfig,
    *,
    batch_size: int,
    oracle: OracleControllerInput | None = None,
    fisher: FisherRepresentation | None = None,
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
    signal_squared = _quadratic_energy(
        state.trend,
        config.risk_metric,
        fisher,
    )
    plugin_pi = fixed_batch_optimal_pi(
        signal_squared,
        trace,
        state.q,
        batch_size,
        epsilon=config.trace_epsilon,
    )
    cold_pi = batch_size / (state.effective_size + batch_size)
    cold_start = state.environment_distance < config.trend_half_life_p
    zero_information = (
        config.risk_metric == "fisher"
        and signal_squared == 0.0
        and state.residual_moment == 0.0
    )
    if zero_information:
        plugin_pi = float(config.pi_min)
    zero_information_fallback = (
        zero_information
        and not cold_start
        and config.policy == "optimal_plugin"
    )

    oracle_pi = None
    theory_pi = None
    oracle_trace = state.oracle_trace_estimate(config.trace_epsilon)
    if oracle is not None:
        oracle_signal_squared = _quadratic_energy(
            oracle.displacement,
            config.risk_metric,
            fisher,
        )
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
        risk_metric=config.risk_metric,
        uncertainty_scale_semantics=(
            "trace_inverse_fisher"
            if config.risk_metric == "euclidean"
            else "fisher_weighted_covariance_dimension"
        ),
        zero_information_fallback=zero_information_fallback,
    )


def advance_discounted_risk(
    state: DiscountedRiskState,
    old_risk: float,
    new_risk: float,
    *,
    half_life_steps: float,
) -> tuple[DiscountedRiskState, float]:
    """Accumulate coefficients of an exponentially discounted quadratic risk."""

    state.validate()
    if (
        not math.isfinite(old_risk)
        or not math.isfinite(new_risk)
        or old_risk < 0.0
        or new_risk < 0.0
    ):
        raise ValueError("risk coefficients must be finite and nonnegative")
    gain = update_step_half_life_gain(half_life_steps)
    updated = DiscountedRiskState(
        old_risk_moment=(1.0 - gain) * state.old_risk_moment + gain * old_risk,
        new_risk_moment=(1.0 - gain) * state.new_risk_moment + gain * new_risk,
        updates=state.updates + 1,
    )
    updated.validate()
    return updated, gain


def decide_discounted_risk_controller(
    state: ControllerState,
    discounted_state: DiscountedRiskState,
    config: ControllerConfig,
    *,
    batch_size: int,
    fisher: FisherRepresentation,
) -> DiscountedRiskDecision:
    """Choose a predictable action from discounted Fisher-risk coefficients."""

    config.validate()
    if config.policy != "discounted_risk":
        raise ValueError("discounted risk decision requires its named policy")
    if config.action_half_life_steps is None:
        raise ValueError("discounted risk decision requires an action half-life")
    if config.pi_min is None or config.trend_half_life_p is None:
        raise ValueError("discounted risk decision requires unified bounds")
    if config.trace_epsilon is None:
        raise ValueError("discounted risk decision requires trace_epsilon")
    if config.risk_metric != "fisher":
        raise ValueError("discounted risk decision requires Fisher risk")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    trace = state.trace_estimate(config.trace_epsilon)
    signal = _quadratic_energy(state.trend, "fisher", fisher)
    old_covariance = trace * state.q
    new_covariance = trace / batch_size
    old_risk = signal + old_covariance
    new_risk = new_covariance
    instantaneous_pi = fixed_batch_optimal_pi(
        signal,
        trace,
        state.q,
        batch_size,
        epsilon=config.trace_epsilon,
    )
    updated, gain = advance_discounted_risk(
        discounted_state,
        old_risk,
        new_risk,
        half_life_steps=config.action_half_life_steps,
    )
    denominator = updated.old_risk_moment + updated.new_risk_moment
    zero_denominator = denominator == 0.0
    unclipped_pi = (
        None if zero_denominator else updated.old_risk_moment / denominator
    )
    cold_start = state.environment_distance < config.trend_half_life_p
    raw_pi = (
        float(config.fixed_pi)
        if cold_start or zero_denominator
        else float(unclipped_pi)
    )
    applied_pi, lower_active, upper_active = _bounded(
        raw_pi,
        float(config.pi_min),
        float(config.pi_max),
    )
    controller = ControllerDecision(
        policy=config.policy,
        raw_pi=raw_pi,
        applied_pi=applied_pi,
        plugin_pi=instantaneous_pi,
        cold_start_pi=float(config.fixed_pi),
        oracle_pi=None,
        theory_oracle_pi=None,
        lower_bound_active=lower_active,
        upper_bound_active=upper_active,
        cold_start_active=cold_start,
        ewc_odds=(1.0 - applied_pi) / applied_pi,
        signal_squared=signal,
        trace_estimate=trace,
        oracle_trace_estimate=None,
        old_covariance_trace=old_covariance,
        new_covariance_trace=new_covariance,
        effective_size=state.effective_size,
        risk_metric="fisher",
        uncertainty_scale_semantics="fisher_weighted_covariance_dimension",
        zero_information_fallback=zero_denominator,
    )
    return DiscountedRiskDecision(
        controller=controller,
        state=updated,
        gain=gain,
        instantaneous_old_risk=old_risk,
        instantaneous_new_risk=new_risk,
        unclipped_pi=unclipped_pi,
        zero_denominator_fallback=zero_denominator,
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
    fisher: FisherRepresentation | None = None,
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
        residual_euclidean_squared = 0.0
        scale = 0.0
        normalized_displacement = None
        oracle_residual = None
        oracle_residual_energy = None
        oracle_residual_euclidean_squared = None
        residual_moment = state.residual_moment
        scale_moment = state.scale_moment
        oracle_residual_moment = state.oracle_residual_moment
        oracle_scale_moment = state.oracle_scale_moment
        trend = state.trend
    else:
        # Locally, the EWC minimizer moves by pi times the unregularized drift.
        normalized_displacement = displacement / pi
        residual = displacement - pi * state.trend
        residual_euclidean_squared = float(residual @ residual)
        residual_squared = _quadratic_energy(
            residual,
            decision.risk_metric,
            fisher,
        )
        scale = pi**2 * (state.q + 1.0 / batch_size)
        residual_moment = (
            (1.0 - gain) * state.residual_moment + gain * residual_squared
        )
        scale_moment = (1.0 - gain) * state.scale_moment + gain * scale
        if oracle_displacement is None:
            oracle_residual = None
            oracle_residual_energy = None
            oracle_residual_euclidean_squared = None
            oracle_residual_moment = state.oracle_residual_moment
            oracle_scale_moment = state.oracle_scale_moment
        else:
            oracle_residual = displacement - pi * oracle_displacement
            oracle_residual_euclidean_squared = float(
                oracle_residual @ oracle_residual
            )
            oracle_residual_energy = _quadratic_energy(
                oracle_residual,
                decision.risk_metric,
                fisher,
            )
            oracle_residual_moment = (
                (1.0 - gain) * state.oracle_residual_moment
                + gain * oracle_residual_energy
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
        residual_euclidean_squared=residual_euclidean_squared,
        oracle_residual_energy=oracle_residual_energy,
        oracle_residual_euclidean_squared=(
            oracle_residual_euclidean_squared
        ),
    )
