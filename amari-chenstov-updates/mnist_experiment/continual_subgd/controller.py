"""Causal innovation controller for adaptive online SubGD."""

from __future__ import annotations

import dataclasses
import math

from .geometry import half_life_gain


@dataclasses.dataclass(frozen=True)
class InnovationControllerConfig:
    innovation_half_life: float
    alpha_scale: float
    beta_min_half_life: float
    beta_max_half_life: float
    beta_scale: float

    def __post_init__(self) -> None:
        for name in (
            "innovation_half_life",
            "alpha_scale",
            "beta_min_half_life",
            "beta_max_half_life",
            "beta_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.beta_max_half_life > self.beta_min_half_life:
            raise ValueError("fast beta half-life cannot exceed slow half-life")

    @property
    def tau(self) -> float:
        return half_life_gain(self.innovation_half_life)

    @property
    def beta_min(self) -> float:
        return half_life_gain(self.beta_min_half_life)

    @property
    def beta_max(self) -> float:
        return half_life_gain(self.beta_max_half_life)


@dataclasses.dataclass(frozen=True)
class InnovationDecision:
    raw_innovation: float
    smoothed_innovation: float
    alpha: float
    beta: float

    def mapping(self) -> dict[str, float]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class InnovationControllerState:
    smoothed_innovation: float = 0.0
    updates: int = 0

    def decide(
        self,
        innovation: float,
        config: InnovationControllerConfig,
    ) -> tuple[InnovationDecision, "InnovationControllerState"]:
        value = float(innovation)
        if not math.isfinite(value) or not 0 <= value <= 1 + 1e-9:
            raise ValueError("innovation must lie in [0, 1]")
        value = min(value, 1.0)
        smoothed = (1 - config.tau) * self.smoothed_innovation + config.tau * value
        alpha = 1.0 / (1.0 + config.alpha_scale * smoothed)
        fraction = config.beta_scale * smoothed / (1.0 + config.beta_scale * smoothed)
        beta = config.beta_min + (config.beta_max - config.beta_min) * fraction
        decision = InnovationDecision(value, smoothed, alpha, beta)
        return decision, InnovationControllerState(smoothed, self.updates + 1)
