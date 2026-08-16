"""Recursive diagonal and low-rank-plus-diagonal Fisher tracking."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import Tensor

from .directional_ridge import DirectionalRidgeLFUState
from .fixed_trajectory import OnlineStepStatistics
from .lanczos_wrapper import (
    LanczosDiagnostics,
    approximate_low_rank_diagonal,
)
from .reference import relative_frobenius_error
from .representations import DiagonalFisher, LowRankDiagonalFisher


STRUCTURED_CORRECTION_METHODS = frozenset({"ema", "ac_only", "full_lfu"})


def _select_correction(method: str, ridge) -> Tensor:
    if method == "ema":
        return torch.zeros_like(ridge.full)
    if method == "ac_only":
        return ridge.amari_chentsov
    return ridge.full


@dataclasses.dataclass(frozen=True)
class DiagonalProjectionDiagnostics:
    pre_projection_minimum: float
    negative_entry_count: int
    negative_mass: float
    projection_distance: float
    relative_projection_distance: float


def _project_diagonal(values: Tensor) -> tuple[Tensor, DiagonalProjectionDiagnostics]:
    negative = values < 0
    projected = values.clamp_min(0)
    distance = torch.linalg.vector_norm(projected - values)
    denominator = torch.linalg.vector_norm(values).clamp_min(
        torch.finfo(values.dtype).eps
    )
    return projected, DiagonalProjectionDiagnostics(
        pre_projection_minimum=float(values.min()),
        negative_entry_count=int(negative.sum()),
        negative_mass=float((-values[negative]).sum()),
        projection_distance=float(distance),
        relative_projection_distance=float(distance / denominator),
    )


@dataclasses.dataclass(frozen=True)
class DiagonalFisherUpdate:
    step: int
    representation: DiagonalFisher
    prediction: Tensor
    candidate: Tensor
    correction: Tensor
    projection: DiagonalProjectionDiagnostics
    ridge_metrics: dict[str, Any]
    ridge_state: dict[str, Any]
    blend_gain: float
    correction_method: str


class DiagonalFisherTracker:
    def __init__(
        self,
        initial_fisher: Tensor,
        *,
        ema_gain: float | None,
        ridge_half_life_steps: float,
        ridge_amplitude_epsilon: float,
        ridge_coherence_threshold: float,
        correction_method: str = "full_lfu",
    ) -> None:
        if ema_gain is not None and not 0.0 < ema_gain <= 1.0:
            raise ValueError("ema_gain must be in (0, 1]")
        if (
            initial_fisher.ndim != 2
            or initial_fisher.shape[0] != initial_fisher.shape[1]
            or not initial_fisher.is_floating_point()
            or not torch.isfinite(initial_fisher).all()
        ):
            raise ValueError("initial_fisher must be a finite floating square matrix")
        initial, _ = _project_diagonal(torch.diagonal(initial_fisher))
        if correction_method not in STRUCTURED_CORRECTION_METHODS:
            raise ValueError(
                f"unsupported structured correction method: {correction_method}"
            )
        self.previous = DiagonalFisher(initial)
        self.ema_gain = None if ema_gain is None else float(ema_gain)
        self.correction_method = correction_method
        self.next_step = 0
        self.ridge = DirectionalRidgeLFUState(
            half_life_steps=ridge_half_life_steps,
            amplitude_epsilon=ridge_amplitude_epsilon,
            coherence_threshold=ridge_coherence_threshold,
        )

    def update(
        self,
        step: int,
        statistics: OnlineStepStatistics,
        direction: Tensor,
        *,
        blend_gain: float | None = None,
    ) -> DiagonalFisherUpdate:
        if step != self.next_step:
            raise ValueError(
                f"expected structured step {self.next_step}, received {step}"
            )
        statistics.validate(self.previous.shape[0])
        gain = self.ema_gain if blend_gain is None else float(blend_gain)
        if gain is None or not 0.0 <= gain <= 1.0:
            raise ValueError(
                "a fixed ema_gain or per-step blend_gain in [0, 1] is required"
            )
        ridge = self.ridge.update(
            direction,
            statistics.estimate.amari_chentsov,
            statistics.estimate.residual,
        )
        if step == 0:
            correction = torch.zeros_like(self.previous.values)
            prediction = self.previous.values
            candidate = self.previous.values
        else:
            correction = torch.diagonal(
                _select_correction(self.correction_method, ridge)
            )
            prediction = self.previous.values + correction
            candidate = (
                (1.0 - gain) * prediction
                + gain
                * torch.diagonal(statistics.estimate.fisher)
            )
        projected, diagnostics = _project_diagonal(candidate)
        self.previous = DiagonalFisher(projected)
        self.next_step += 1
        return DiagonalFisherUpdate(
            step=step,
            representation=self.previous,
            prediction=prediction,
            candidate=candidate,
            correction=correction,
            projection=diagnostics,
            ridge_metrics=ridge.metrics_mapping(),
            ridge_state=ridge.state_artifact(),
            blend_gain=gain,
            correction_method=self.correction_method,
        )


@dataclasses.dataclass(frozen=True)
class LowRankDiagonalUpdate:
    step: int
    representation: LowRankDiagonalFisher
    candidate: Tensor
    correction: Tensor
    candidate_minimum_eigenvalue: float
    candidate_negative_eigenvalue_count: int
    candidate_materially_negative_eigenvalue_count: int
    candidate_spectral_tolerance: float
    candidate_numerical_rank: int
    candidate_eigendecomposition_backend: str
    lanczos: LanczosDiagnostics
    ridge_metrics: dict[str, Any]
    ridge_state: dict[str, Any]
    blend_gain: float
    correction_method: str


@dataclasses.dataclass(frozen=True)
class _CandidateSpectrumDiagnostics:
    minimum_eigenvalue: float
    negative_eigenvalue_count: int
    materially_negative_eigenvalue_count: int
    tolerance: float
    numerical_rank: int
    backend: str


def _candidate_spectrum_diagnostics(
    candidate: Tensor,
) -> _CandidateSpectrumDiagnostics:
    """Diagnose a symmetric candidate without making CUDA convergence fatal."""

    backend = candidate.device.type
    try:
        eigenvalues = torch.linalg.eigvalsh(candidate)
    except RuntimeError as error:
        message = str(error)
        if "linalg.eigh" not in message or "failed to converge" not in message:
            raise
        eigenvalues = torch.linalg.eigvalsh(
            candidate.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
        backend = "cpu_float64_fallback"

    spectral_scale = max(float(eigenvalues.abs().max()), 1.0)
    tolerance = (
        torch.finfo(eigenvalues.dtype).eps
        * candidate.shape[0]
        * spectral_scale
    )
    return _CandidateSpectrumDiagnostics(
        minimum_eigenvalue=float(eigenvalues.min()),
        negative_eigenvalue_count=int((eigenvalues < 0).sum()),
        materially_negative_eigenvalue_count=int(
            (eigenvalues < -tolerance).sum()
        ),
        tolerance=float(tolerance),
        numerical_rank=int((eigenvalues.abs() > tolerance).sum()),
        backend=backend,
    )


class LowRankDiagonalFisherTracker:
    def __init__(
        self,
        initial_fisher: Tensor,
        *,
        rank: int,
        ema_gain: float | None,
        ridge_half_life_steps: float,
        ridge_amplitude_epsilon: float,
        ridge_coherence_threshold: float,
        correction_method: str = "full_lfu",
    ) -> None:
        if ema_gain is not None and not 0.0 < ema_gain <= 1.0:
            raise ValueError("ema_gain must be in (0, 1]")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        if (
            initial_fisher.ndim != 2
            or initial_fisher.shape[0] != initial_fisher.shape[1]
            or not initial_fisher.is_floating_point()
            or not torch.isfinite(initial_fisher).all()
            or rank > initial_fisher.shape[0]
        ):
            raise ValueError(
                "initial_fisher must be finite, floating, square, and support rank"
            )
        self.initial_fisher = initial_fisher
        if correction_method not in STRUCTURED_CORRECTION_METHODS:
            raise ValueError(
                f"unsupported structured correction method: {correction_method}"
            )
        self.previous: LowRankDiagonalFisher | None = None
        self.rank = rank
        self.ema_gain = None if ema_gain is None else float(ema_gain)
        self.correction_method = correction_method
        self.next_step = 0
        self.ridge = DirectionalRidgeLFUState(
            half_life_steps=ridge_half_life_steps,
            amplitude_epsilon=ridge_amplitude_epsilon,
            coherence_threshold=ridge_coherence_threshold,
        )

    def update(
        self,
        step: int,
        statistics: OnlineStepStatistics,
        direction: Tensor,
        *,
        lanczos_seed: int,
        blend_gain: float | None = None,
    ) -> LowRankDiagonalUpdate:
        if step != self.next_step:
            raise ValueError(
                f"expected structured step {self.next_step}, received {step}"
            )
        parameter_count = self.initial_fisher.shape[0]
        statistics.validate(parameter_count)
        gain = self.ema_gain if blend_gain is None else float(blend_gain)
        if gain is None or not 0.0 <= gain <= 1.0:
            raise ValueError(
                "a fixed ema_gain or per-step blend_gain in [0, 1] is required"
            )
        ridge = self.ridge.update(
            direction,
            statistics.estimate.amari_chentsov,
            statistics.estimate.residual,
        )
        if step == 0:
            correction = torch.zeros_like(self.initial_fisher)
            candidate = self.initial_fisher
        else:
            correction = _select_correction(self.correction_method, ridge)
            prediction = self.previous.to_dense() + correction
            candidate = (
                (1.0 - gain) * prediction
                + gain * statistics.estimate.fisher
            )
        candidate = (candidate + candidate.mT) / 2
        spectrum = _candidate_spectrum_diagnostics(candidate)
        krylov_rank_limit = min(
            self.rank,
            spectrum.numerical_rank
            + int(spectrum.numerical_rank < parameter_count),
        )
        approximation = approximate_low_rank_diagonal(
            lambda vector: candidate @ vector,
            torch.diagonal(candidate),
            rank=self.rank,
            seed=lanczos_seed,
            maximum_krylov_rank=krylov_rank_limit,
        )
        self.previous = approximation.representation
        self.next_step += 1
        return LowRankDiagonalUpdate(
            step=step,
            representation=self.previous,
            candidate=candidate,
            correction=correction,
            candidate_minimum_eigenvalue=spectrum.minimum_eigenvalue,
            candidate_negative_eigenvalue_count=(
                spectrum.negative_eigenvalue_count
            ),
            candidate_materially_negative_eigenvalue_count=(
                spectrum.materially_negative_eigenvalue_count
            ),
            candidate_spectral_tolerance=spectrum.tolerance,
            candidate_numerical_rank=spectrum.numerical_rank,
            candidate_eigendecomposition_backend=spectrum.backend,
            lanczos=approximation.diagnostics,
            ridge_metrics=ridge.metrics_mapping(),
            ridge_state=ridge.state_artifact(),
            blend_gain=gain,
            correction_method=self.correction_method,
        )


def structured_representation_metrics(
    representation: DiagonalFisher | LowRankDiagonalFisher,
    dense_target: Tensor,
    reference: Tensor,
    probes: Tensor,
    direction: Tensor,
) -> dict[str, Any]:
    explicit = representation.to_dense()
    dense_products = dense_target @ probes
    represented_products = representation.matvec(probes)
    product_denominator = torch.linalg.matrix_norm(
        dense_products,
        ord="fro",
    ).clamp_min(torch.finfo(dense_target.dtype).eps)
    dense_quadratics = torch.sum(probes * dense_products, dim=0)
    represented_quadratics = torch.stack(
        [
            representation.quadratic(probes[:, index])
            for index in range(probes.shape[1])
        ]
    )
    quadratic_denominator = dense_quadratics.abs().clamp_min(
        torch.finfo(dense_target.dtype).eps
    )
    dense_eigenvalues, dense_eigenvectors = torch.linalg.eigh(dense_target)
    represented_eigenvalues, represented_eigenvectors = torch.linalg.eigh(
        explicit
    )
    dense_directional = direction @ (dense_target @ direction)
    represented_directional = representation.quadratic(direction)
    directional_denominator = torch.abs(dense_directional).clamp_min(
        torch.finfo(dense_target.dtype).eps
    )
    return {
        "relative_frobenius_error_to_dense": relative_frobenius_error(
            explicit,
            dense_target,
        ),
        "relative_frobenius_error_to_reference": relative_frobenius_error(
            explicit,
            reference,
        ),
        "fixed_probe_matvec_relative_error": float(
            torch.linalg.matrix_norm(
                represented_products - dense_products,
                ord="fro",
            )
            / product_denominator
        ),
        "mean_probe_quadratic_relative_error": float(
            torch.mean(
                torch.abs(represented_quadratics - dense_quadratics)
                / quadratic_denominator
            )
        ),
        "maximum_probe_quadratic_relative_error": float(
            torch.max(
                torch.abs(represented_quadratics - dense_quadratics)
                / quadratic_denominator
            )
        ),
        "directional_quadratic_relative_error": float(
            torch.abs(represented_directional - dense_directional)
            / directional_denominator
        ),
        "leading_eigenvalue_relative_error": float(
            torch.abs(
                represented_eigenvalues[-1] - dense_eigenvalues[-1]
            )
            / dense_eigenvalues[-1].abs().clamp_min(
                torch.finfo(dense_target.dtype).eps
            )
        ),
        "leading_eigenvector_alignment": float(
            torch.abs(
                represented_eigenvectors[:, -1]
                @ dense_eigenvectors[:, -1]
            )
        ),
        "representation_storage_bytes": representation.storage_bytes(),
        "dense_storage_bytes": dense_target.numel()
        * dense_target.element_size(),
    }
