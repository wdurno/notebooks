"""Amplitude-safe directional ridge smoothing for noisy LFU corrections."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch
from torch import Tensor


@dataclasses.dataclass(frozen=True)
class DirectionalRidgeUpdate:
    amari_chentsov: Tensor
    residual: Tensor
    direction_norm: float
    signed_amplitude: float
    coherence_before_reset: float | None
    orthogonal_ratio_before_reset: float | None
    no_motion: bool
    cold_started: bool
    direction_reset: bool
    segment_eligible_steps: int
    warmup_mass: float
    prior_mass: float
    amplitude_denominator: float
    regularized_denominator: float
    ridge_leverage: float
    raw_ac_fro: float
    raw_residual_fro: float
    smoothed_ac_fro: float
    smoothed_residual_fro: float
    reference_direction: Tensor | None
    ac_numerator: Tensor | None
    residual_numerator: Tensor | None

    @property
    def full(self) -> Tensor:
        return self.amari_chentsov + self.residual

    def metrics_mapping(self) -> dict[str, Any]:
        return {
            "direction_norm": self.direction_norm,
            "signed_amplitude": self.signed_amplitude,
            "coherence_before_reset": self.coherence_before_reset,
            "orthogonal_ratio_before_reset": (
                self.orthogonal_ratio_before_reset
            ),
            "no_motion": self.no_motion,
            "cold_started": self.cold_started,
            "direction_reset": self.direction_reset,
            "segment_eligible_steps": self.segment_eligible_steps,
            "warmup_mass": self.warmup_mass,
            "prior_mass": self.prior_mass,
            "amplitude_denominator": self.amplitude_denominator,
            "regularized_denominator": self.regularized_denominator,
            "ridge_leverage": self.ridge_leverage,
            "raw_ac_fro": self.raw_ac_fro,
            "raw_residual_fro": self.raw_residual_fro,
            "smoothed_ac_fro": self.smoothed_ac_fro,
            "smoothed_residual_fro": self.smoothed_residual_fro,
            "smoothed_full_fro": float(
                torch.linalg.matrix_norm(self.full, ord="fro")
            ),
        }

    def state_artifact(self) -> dict[str, Any] | None:
        if self.reference_direction is None:
            return None
        return {
            "reference_direction": self.reference_direction.detach().cpu(),
            "ac_numerator": self.ac_numerator.detach().cpu(),
            "residual_numerator": self.residual_numerator.detach().cpu(),
            "amplitude_denominator": self.amplitude_denominator,
            "segment_eligible_steps": self.segment_eligible_steps,
            "warmup_mass": self.warmup_mass,
            "prior_mass": self.prior_mass,
        }


class DirectionalRidgeLFUState:
    """Exponentially weighted ridge state for one locally coherent direction."""

    def __init__(
        self,
        *,
        half_life_steps: float,
        amplitude_epsilon: float,
        coherence_threshold: float,
    ) -> None:
        if not math.isfinite(half_life_steps) or half_life_steps <= 0:
            raise ValueError("half_life_steps must be positive and finite")
        if not math.isfinite(amplitude_epsilon) or amplitude_epsilon <= 0:
            raise ValueError("amplitude_epsilon must be positive and finite")
        if (
            not math.isfinite(coherence_threshold)
            or not 0.0 < coherence_threshold <= 1.0
        ):
            raise ValueError("coherence_threshold must be in (0, 1]")
        self.half_life_steps = float(half_life_steps)
        self.amplitude_epsilon = float(amplitude_epsilon)
        self.coherence_threshold = float(coherence_threshold)
        self.rho = 2.0 ** (-1.0 / self.half_life_steps)
        self.beta = 1.0 - self.rho
        self.reference_direction: Tensor | None = None
        self.ac_numerator: Tensor | None = None
        self.residual_numerator: Tensor | None = None
        self.amplitude_denominator: Tensor | None = None
        self.segment_eligible_steps = 0
        self.warmup_mass = 0.0
        self.prior_mass = 0.0

    @staticmethod
    def _validate_inputs(
        direction: Tensor,
        amari_chentsov: Tensor,
        residual: Tensor,
    ) -> None:
        if direction.ndim != 1:
            raise ValueError("direction must be a vector")
        expected = (direction.numel(), direction.numel())
        if amari_chentsov.shape != expected or residual.shape != expected:
            raise ValueError("LFU matrices must match the direction dimension")
        if (
            amari_chentsov.device != direction.device
            or residual.device != direction.device
            or amari_chentsov.dtype != direction.dtype
            or residual.dtype != direction.dtype
        ):
            raise ValueError("direction and LFU matrices must share dtype/device")
        if not (
            torch.isfinite(direction).all()
            and torch.isfinite(amari_chentsov).all()
            and torch.isfinite(residual).all()
        ):
            raise ValueError("direction and LFU matrices must be finite")

    def _start_segment(
        self,
        unit_direction: Tensor,
        amplitude: Tensor,
        matrix_template: Tensor,
    ) -> None:
        self.reference_direction = unit_direction.detach().clone()
        self.ac_numerator = torch.zeros_like(matrix_template)
        self.residual_numerator = torch.zeros_like(matrix_template)
        self.amplitude_denominator = amplitude.square().detach().clone()
        self.segment_eligible_steps = 0
        self.warmup_mass = 0.0
        self.prior_mass = 1.0

    def _decay_without_observation(self) -> None:
        if self.reference_direction is None:
            return
        self.ac_numerator.mul_(self.rho)
        self.residual_numerator.mul_(self.rho)
        self.amplitude_denominator.mul_(self.rho)
        self.warmup_mass *= self.rho
        self.prior_mass *= self.rho

    def update(
        self,
        direction: Tensor,
        amari_chentsov: Tensor,
        residual: Tensor,
    ) -> DirectionalRidgeUpdate:
        self._validate_inputs(direction, amari_chentsov, residual)
        direction_norm_tensor = torch.linalg.vector_norm(direction)
        direction_norm = float(direction_norm_tensor)
        raw_ac_fro = float(
            torch.linalg.matrix_norm(amari_chentsov, ord="fro")
        )
        raw_residual_fro = float(
            torch.linalg.matrix_norm(residual, ord="fro")
        )
        zero = torch.zeros_like(amari_chentsov)

        if direction_norm <= self.amplitude_epsilon:
            self._decay_without_observation()
            denominator = (
                0.0
                if self.amplitude_denominator is None
                else float(self.amplitude_denominator)
            )
            regularized = denominator + self.amplitude_epsilon**2
            return DirectionalRidgeUpdate(
                amari_chentsov=zero,
                residual=zero.clone(),
                direction_norm=direction_norm,
                signed_amplitude=0.0,
                coherence_before_reset=None,
                orthogonal_ratio_before_reset=None,
                no_motion=True,
                cold_started=False,
                direction_reset=False,
                segment_eligible_steps=self.segment_eligible_steps,
                warmup_mass=self.warmup_mass,
                prior_mass=self.prior_mass,
                amplitude_denominator=denominator,
                regularized_denominator=regularized,
                ridge_leverage=(
                    0.0 if regularized == 0.0 else denominator / regularized
                ),
                raw_ac_fro=raw_ac_fro,
                raw_residual_fro=raw_residual_fro,
                smoothed_ac_fro=0.0,
                smoothed_residual_fro=0.0,
                reference_direction=(
                    None
                    if self.reference_direction is None
                    else self.reference_direction.detach().clone()
                ),
                ac_numerator=(
                    None
                    if self.ac_numerator is None
                    else self.ac_numerator.detach().clone()
                ),
                residual_numerator=(
                    None
                    if self.residual_numerator is None
                    else self.residual_numerator.detach().clone()
                ),
            )

        unit_direction = direction / direction_norm_tensor
        cold_started = self.reference_direction is None
        direction_reset = False
        coherence = None
        orthogonal_ratio = None
        if cold_started:
            amplitude = direction_norm_tensor
            self._start_segment(
                unit_direction,
                amplitude,
                amari_chentsov,
            )
        else:
            amplitude = self.reference_direction @ direction
            coherence_tensor = amplitude.abs() / direction_norm_tensor
            coherence = float(coherence_tensor)
            orthogonal_ratio = math.sqrt(max(0.0, 1.0 - coherence**2))
            if coherence < self.coherence_threshold:
                direction_reset = True
                amplitude = direction_norm_tensor
                self._start_segment(
                    unit_direction,
                    amplitude,
                    amari_chentsov,
                )

        self.ac_numerator.mul_(self.rho).add_(
            self.beta * amplitude * amari_chentsov
        )
        self.residual_numerator.mul_(self.rho).add_(
            self.beta * amplitude * residual
        )
        self.amplitude_denominator.mul_(self.rho).add_(
            self.beta * amplitude.square()
        )
        self.segment_eligible_steps += 1
        self.warmup_mass = self.rho * self.warmup_mass + self.beta
        self.prior_mass *= self.rho

        regularized_denominator = (
            self.amplitude_denominator + self.amplitude_epsilon**2
        )
        smoothed_ac = (
            amplitude * self.ac_numerator / regularized_denominator
        )
        smoothed_residual = (
            amplitude * self.residual_numerator / regularized_denominator
        )
        denominator = float(self.amplitude_denominator)
        regularized = float(regularized_denominator)
        return DirectionalRidgeUpdate(
            amari_chentsov=smoothed_ac,
            residual=smoothed_residual,
            direction_norm=direction_norm,
            signed_amplitude=float(amplitude),
            coherence_before_reset=coherence,
            orthogonal_ratio_before_reset=orthogonal_ratio,
            no_motion=False,
            cold_started=cold_started,
            direction_reset=direction_reset,
            segment_eligible_steps=self.segment_eligible_steps,
            warmup_mass=self.warmup_mass,
            prior_mass=self.prior_mass,
            amplitude_denominator=denominator,
            regularized_denominator=regularized,
            ridge_leverage=denominator / regularized,
            raw_ac_fro=raw_ac_fro,
            raw_residual_fro=raw_residual_fro,
            smoothed_ac_fro=float(
                torch.linalg.matrix_norm(smoothed_ac, ord="fro")
            ),
            smoothed_residual_fro=float(
                torch.linalg.matrix_norm(smoothed_residual, ord="fro")
            ),
            reference_direction=self.reference_direction.detach().clone(),
            ac_numerator=self.ac_numerator.detach().clone(),
            residual_numerator=self.residual_numerator.detach().clone(),
        )
