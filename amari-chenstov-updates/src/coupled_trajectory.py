"""Dense Fisher tracking state for EWC-coupled parameter trajectories."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import Tensor

from .directional_ridge import DirectionalRidgeLFUState
from .fixed_trajectory import (
    DENSE_METHODS,
    RIDGE_DENSE_METHODS,
    OnlineStepStatistics,
)
from .reference import relative_frobenius_error
from .representations import (
    PSDProjectionDiagnostics,
    project_psd_frobenius,
)

COUPLED_DENSE_METHODS = DENSE_METHODS + RIDGE_DENSE_METHODS


@dataclasses.dataclass(frozen=True)
class DenseFisherUpdate:
    method: str
    step: int
    estimate: Tensor
    prediction: Tensor
    candidate: Tensor
    correction: Tensor
    fresh_replacement: bool
    projection: PSDProjectionDiagnostics
    ridge_metrics: dict[str, Any] | None
    ridge_state: dict[str, Any] | None


class DenseFisherTracker:
    """Advance one dense estimator along the parameter path it helps create."""

    def __init__(
        self,
        method: str,
        initial_fisher: Tensor,
        *,
        ema_gain: float,
        fresh_fisher_cadence: int,
        ridge_half_life_steps: float | None = None,
        ridge_amplitude_epsilon: float | None = None,
        ridge_coherence_threshold: float | None = None,
    ) -> None:
        if method not in COUPLED_DENSE_METHODS:
            raise ValueError(f"unsupported dense Fisher method: {method}")
        if not 0.0 < ema_gain <= 1.0:
            raise ValueError("ema_gain must be in (0, 1]")
        if fresh_fisher_cadence < 1:
            raise ValueError("fresh_fisher_cadence must be positive")
        ridge_settings = (
            ridge_half_life_steps,
            ridge_amplitude_epsilon,
            ridge_coherence_threshold,
        )
        if method in RIDGE_DENSE_METHODS and any(
            value is None for value in ridge_settings
        ):
            raise ValueError("ridge methods require all ridge settings")

        self.method = method
        self.ema_gain = float(ema_gain)
        self.fresh_fisher_cadence = int(fresh_fisher_cadence)
        self.previous = project_psd_frobenius(initial_fisher).projected
        self.next_step = 0
        self.ridge = (
            DirectionalRidgeLFUState(
                half_life_steps=float(ridge_half_life_steps),
                amplitude_epsilon=float(ridge_amplitude_epsilon),
                coherence_threshold=float(ridge_coherence_threshold),
            )
            if method in RIDGE_DENSE_METHODS
            else None
        )

    def update(
        self,
        step: int,
        statistics: OnlineStepStatistics,
        direction: Tensor,
        fresh_reference: Tensor,
    ) -> DenseFisherUpdate:
        if step != self.next_step:
            raise ValueError(
                f"expected coupled step {self.next_step}, received {step}"
            )
        parameter_count = self.previous.shape[0]
        statistics.validate(parameter_count)
        if direction.shape != (parameter_count,):
            raise ValueError("direction has an invalid shape")
        if fresh_reference.shape != self.previous.shape:
            raise ValueError("fresh reference has an invalid shape")
        if (
            direction.device != self.previous.device
            or direction.dtype != self.previous.dtype
            or fresh_reference.device != self.previous.device
            or fresh_reference.dtype != self.previous.dtype
        ):
            raise ValueError("tracker inputs must share dtype and device")

        ridge_update = (
            None
            if self.ridge is None
            else self.ridge.update(
                direction,
                statistics.estimate.amari_chentsov,
                statistics.estimate.residual,
            )
        )
        if step == 0:
            correction = torch.zeros_like(self.previous)
            prediction = self.previous
            candidate = self.previous
            refreshed = False
        else:
            if self.method in {"ema", "periodic_fresh"}:
                correction = torch.zeros_like(self.previous)
            elif self.method == "ac_only":
                correction = statistics.estimate.amari_chentsov
            elif self.method == "full_lfu":
                correction = statistics.estimate.full
            elif self.method == "ridge_ac_only":
                correction = ridge_update.amari_chentsov
            else:
                correction = ridge_update.full
            prediction = self.previous + correction
            candidate = (
                (1.0 - self.ema_gain) * prediction
                + self.ema_gain * statistics.estimate.fisher
            )
            refreshed = (
                self.method == "periodic_fresh"
                and step % self.fresh_fisher_cadence == 0
            )
            if refreshed:
                candidate = fresh_reference

        projected = project_psd_frobenius(candidate)
        self.previous = projected.projected
        self.next_step += 1
        return DenseFisherUpdate(
            method=self.method,
            step=step,
            estimate=self.previous,
            prediction=prediction,
            candidate=candidate,
            correction=correction,
            fresh_replacement=refreshed,
            projection=projected.diagnostics,
            ridge_metrics=(
                None if ridge_update is None else ridge_update.metrics_mapping()
            ),
            ridge_state=(
                None if ridge_update is None else ridge_update.state_artifact()
            ),
        )


def dense_fisher_metrics(
    update: DenseFisherUpdate,
    reference: Tensor,
    direction: Tensor,
) -> dict[str, Any]:
    """Return scalar diagnostics against the reference at this exact path point."""

    error = update.estimate - reference
    reference_eigenvalues, reference_eigenvectors = torch.linalg.eigh(reference)
    estimate_eigenvalues, estimate_eigenvectors = torch.linalg.eigh(
        update.estimate
    )
    directional_error = direction @ (error @ direction)
    return {
        "relative_frobenius_error": relative_frobenius_error(
            update.estimate,
            reference,
        ),
        "prediction_relative_frobenius_error": relative_frobenius_error(
            update.prediction,
            reference,
        ),
        "candidate_relative_frobenius_error": relative_frobenius_error(
            update.candidate,
            reference,
        ),
        "absolute_frobenius_error": float(
            torch.linalg.matrix_norm(error, ord="fro")
        ),
        "operator_norm_error": float(
            torch.linalg.eigvalsh((error + error.mT) / 2).abs().max()
        ),
        "directional_error": float(directional_error),
        "absolute_directional_error": float(torch.abs(directional_error)),
        "estimate_fro": float(
            torch.linalg.matrix_norm(update.estimate, ord="fro")
        ),
        "applied_correction_fro": float(
            torch.linalg.matrix_norm(update.correction, ord="fro")
        ),
        "leading_eigenvalue": float(estimate_eigenvalues[-1]),
        "reference_leading_eigenvalue": float(reference_eigenvalues[-1]),
        "leading_eigenvector_alignment": float(
            torch.abs(
                estimate_eigenvectors[:, -1]
                @ reference_eigenvectors[:, -1]
            )
        ),
        "fresh_replacement": update.fresh_replacement,
        "ridge": update.ridge_metrics,
        "projection": dataclasses.asdict(update.projection),
    }
