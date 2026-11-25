"""Bounded Fix5 Fisher archive with an exact full diagonal and thin factors."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import Tensor

from src.representations import LowRankDiagonalFisher


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


class Archive:
    """Represent F = LL^T + diag(d - rowsum(L**2)), not diag(d) + LL^T.

    Fresh columns are already normalized likelihood-score (or analytic Fisher)
    factors. Their count does not change the evidence weights alpha and beta.
    Compression is streaming and does not recover correlations lost to the
    residual diagonal. Fresh columns are processed in their supplied order.
    """

    FRESH_CHUNK_COLUMNS = 32

    def __init__(self, diagonal: Tensor, factor: Tensor | None = None, rank: int = 0):
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a nonnegative integer")
        if (
            diagonal.ndim != 1
            or diagonal.numel() == 0
            or diagonal.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(diagonal).all()
            or bool((diagonal < 0).any())
        ):
            raise ValueError("diagonal must be a finite nonnegative float32/64 vector")
        self.rank = rank
        self._diagonal = diagonal.detach().clone()
        if factor is None:
            factor = diagonal.new_empty((diagonal.numel(), 0))
        self._validate(factor, matrix=True)
        if factor.shape[1] > rank:
            raise ValueError("factor width exceeds the configured rank")
        factor, integrity = self._correct_roundoff(self.diagonal, factor.detach().clone())
        self._representation = LowRankDiagonalFisher(factor, self.diagonal - factor.square().sum(1))
        self._update_diagnostics: dict[str, Any] = integrity
        self._solve_cache: dict[str, Any] | None = None

    @property
    def diagonal(self) -> Tensor:
        """The full represented diagonal d, including the factor contribution."""
        return self._diagonal

    @property
    def factor(self) -> Tensor:
        return self._representation.factor

    @property
    def residual(self) -> Tensor:
        return self._representation.residual_diagonal

    @property
    def residual_diagonal(self) -> Tensor:
        return self.residual

    def _validate(self, value: Tensor, *, matrix: bool = False) -> None:
        if (
            value.ndim not in ((2,) if matrix else (1, 2))
            or value.shape[0] != self.diagonal.numel()
            or value.device != self.diagonal.device
            or value.dtype != self.diagonal.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("tensor must have matching parameter dimension, dtype/device and finite values")

    @staticmethod
    def _correct_roundoff(diagonal: Tensor, factor: Tensor) -> tuple[Tensor, dict[str, Any]]:
        energy = factor.square().sum(1)
        residual = diagonal - energy
        if not torch.isfinite(energy).all():
            raise ValueError("factor row energy overflowed")
        integrity = {
            "minimum_residual_before_correction": float(residual.min()),
            "negative_residual_mass_before_correction": float((-residual.clamp_max(0)).sum()),
            "roundoff_corrected_rows": int((residual < 0).sum()),
        }
        eps = torch.finfo(diagonal.dtype).eps
        tolerance = 32 * eps * torch.maximum(diagonal, energy)
        if bool((residual < -tolerance).any()):
            raise ValueError(f"material negative residual: {integrity}")
        if bool((residual < 0).any()):
            # Correct only roundoff, by shrinking L rather than increasing d.
            bad = residual < 0
            factor = factor.clone()
            scale = (diagonal[bad] / energy[bad]).sqrt() * (1 - 2 * eps)
            factor[bad] *= scale.unsqueeze(1)
        return factor, integrity

    @torch.no_grad()
    def update(self, fresh_factor: Tensor, alpha: float, beta: float) -> dict[str, Any]:
        """Mix evidence and truncate bounded thin SVDs, preserving d and trace.

        The alpha scaling is applied once, not once per fresh chunk. Diagnostics
        describe local truncations, not accumulated historical Fisher error.
        Failed validation/compression leaves the existing archive unchanged.
        """
        self._validate(fresh_factor, matrix=True)
        alpha = _nonnegative(alpha, "alpha")
        beta = _nonnegative(beta, "beta")
        diagonal = alpha * self.diagonal + beta * fresh_factor.square().sum(1)
        if not torch.isfinite(diagonal).all():
            raise ValueError("updated diagonal overflowed")
        factor = math.sqrt(alpha) * self.factor
        tail_trace = 0.0
        largest_discarded = 0.0
        chunk_sizes = []
        if self.rank and beta:
            for chunk in fresh_factor.split(self.FRESH_CHUNK_COLUMNS, dim=1):
                if chunk.shape[1] == 0:
                    continue
                chunk_sizes.append(chunk.shape[1])
                augmented = torch.cat((factor, math.sqrt(beta) * chunk), dim=1)
                # No parameter-square matrix or inverse of weak singular values.
                _, singular, right = torch.linalg.svd(augmented, full_matrices=False)
                retained = min(self.rank, singular.numel())
                discarded = singular[retained:].square()
                tail_trace += float(discarded.sum())
                if discarded.numel():
                    largest_discarded = max(largest_discarded, float(discarded[0]))
                # AV_r also preserves exactly zero rows in degenerate factors.
                factor = augmented @ right[:retained].mT
        elif not self.rank:
            # Rank zero is exactly the diagonal control; never perform an SVD.
            tail_trace = float((beta * fresh_factor.square().sum(1)).sum())
            # Computing the largest discarded value would defeat this fast path.
            largest_discarded = None
        factor, integrity = self._correct_roundoff(diagonal, factor)
        representation = LowRankDiagonalFisher(factor, diagonal - factor.square().sum(1))
        self._diagonal = diagonal
        self._representation = representation
        self._solve_cache = None
        self._update_diagnostics = {
            **integrity,
            "alpha": alpha,
            "beta": beta,
            "fresh_columns": fresh_factor.shape[1],
            "chunk_sizes": chunk_sizes,
            "chunk_columns_limit": self.FRESH_CHUNK_COLUMNS,
            "compression_tail_trace": tail_trace,
            "largest_discarded_squared_singular_value": largest_discarded,
        }
        return self.diagnostics()

    def quadratic(self, vector: Tensor) -> Tensor:
        self._validate(vector)
        return self._representation.quadratic(vector)

    def matvec(self, vector: Tensor) -> Tensor:
        return self._representation.matvec(vector)

    def solve(self, rhs: Tensor, damping: Tensor, weight: float = 1) -> Tensor:
        """Solve (diag(damping) + weight F)x = rhs via cached Woodbury.

        Damping is fixed numerical state; gradients with respect to rhs remain
        available. Vector and multiple-column right hand sides are supported.
        """
        self._validate(rhs)
        self._validate(damping)
        if damping.ndim != 1 or bool((damping <= 0).any()):
            raise ValueError("damping must be a strictly positive parameter vector")
        weight = _nonnegative(weight, "weight")
        cache = self._solve_cache
        if cache is None or cache["weight"] != weight or not torch.equal(cache["damping"], damping):
            with torch.no_grad():
                denominator = damping.detach() + weight * self.residual
                if not torch.isfinite(denominator).all():
                    raise ValueError("damped diagonal overflowed")
                inverse = denominator.reciprocal()
                weighted_factor = math.sqrt(weight) * self.factor
                scaled_factor = inverse.unsqueeze(1) * weighted_factor
                cholesky = None
                if weight and self.factor.shape[1]:
                    gram = weighted_factor.mT @ scaled_factor
                    middle = (gram + gram.mT) / 2 + torch.eye(
                        gram.shape[0], dtype=gram.dtype, device=gram.device
                    )
                    cholesky = torch.linalg.cholesky(middle)
                cache = {
                    "weight": weight,
                    "damping": damping.detach().clone(),
                    "inverse": inverse,
                    "weighted_factor": weighted_factor,
                    "scaled_factor": scaled_factor,
                    "cholesky": cholesky,
                }
                self._solve_cache = cache
        matrix_rhs = rhs.unsqueeze(1) if rhs.ndim == 1 else rhs
        result = cache["inverse"].unsqueeze(1) * matrix_rhs
        if cache["cholesky"] is not None:
            result = result - cache["scaled_factor"] @ torch.cholesky_solve(
                cache["weighted_factor"].mT @ result, cache["cholesky"]
            )
        return result.squeeze(1) if rhs.ndim == 1 else result

    def diagnostics(self) -> dict[str, Any]:
        """JSON-ready factor spectrum, not the full archive eigenspectrum."""
        with torch.no_grad():
            gram = self.factor.mT @ self.factor
            eigenvalues = torch.linalg.eigvalsh((gram + gram.mT) / 2).flip(0)
            trace = float(self.diagonal.sum())
            correlated_trace = float(self.factor.square().sum())
            represented = self.factor.square().sum(1) + self.residual
            threshold = (
                float(eigenvalues[0].clamp_min(0))
                * max(self.factor.shape)
                * torch.finfo(self.diagonal.dtype).eps
                if eigenvalues.numel() else 0.0
            )
            return {
                **copy.deepcopy(self._update_diagnostics),
                "rank": self.rank,
                "factor_columns": self.factor.shape[1],
                "realized_rank": int((eigenvalues > threshold).sum()),
                "factor_eigenvalues": eigenvalues.cpu().tolist(),
                "trace": trace,
                "correlated_trace": correlated_trace,
                "correlated_trace_fraction": correlated_trace / trace if trace else None,
                "residual_trace": float(self.residual.sum()),
                "minimum_residual": float(self.residual.min()),
                "diagonal_max_abs_error": float((represented - self.diagonal).abs().max()),
                "trace_error": float(represented.sum() - self.diagonal.sum()),
                "storage_bytes": self._representation.storage_bytes() + self.diagonal.numel() * self.diagonal.element_size(),
            }

    def state_dict(self) -> dict[str, Any]:
        return {
            "diagonal": self.diagonal.detach().cpu().clone(),
            "factor": self.factor.detach().cpu().clone(),
            "rank": self.rank,
            "update_diagnostics": copy.deepcopy(self._update_diagnostics),
        }

    @classmethod
    def from_state_dict(
        cls, state: dict[str, Any], device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
    ) -> Archive:
        archive = cls(
            state["diagonal"].to(device=device, dtype=dtype),
            state["factor"].to(device=device, dtype=dtype),
            rank=state["rank"],
        )
        if "update_diagnostics" in state:
            archive._update_diagnostics = copy.deepcopy(state["update_diagnostics"])
        return archive
