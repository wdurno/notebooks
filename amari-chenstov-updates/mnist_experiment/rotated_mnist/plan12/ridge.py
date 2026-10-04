"""Ridge geometries and diagnostics for Plan 12."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch import Tensor

from src.representations import FisherRepresentation


FisherLike = Tensor | FisherRepresentation


def _matvec(fisher: FisherLike, vector: Tensor) -> Tensor:
    return fisher @ vector if isinstance(fisher, Tensor) else fisher.matvec(vector)


def _quadratic(fisher: FisherLike, vector: Tensor) -> Tensor:
    return vector @ (fisher @ vector) if isinstance(fisher, Tensor) else fisher.quadratic(vector)


def _diagonal(fisher: FisherLike) -> Tensor:
    return torch.diagonal(fisher) if isinstance(fisher, Tensor) else fisher.diagonal_vector()


@dataclasses.dataclass(frozen=True)
class RidgeFisher:
    """A Fisher operator plus isotropic or unresolved-subspace ridge."""

    base: FisherLike
    kappa: float
    geometry: str = "isotropic"
    resolved_basis: Tensor | None = None

    def __post_init__(self) -> None:
        if self.geometry not in {"isotropic", "tail"}:
            raise ValueError("ridge geometry must be isotropic or tail")
        if not torch.isfinite(torch.tensor(self.kappa)) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")
        if self.geometry == "tail" and self.resolved_basis is None:
            raise ValueError("tail ridge requires a resolved basis")
        if self.resolved_basis is not None:
            basis = self.resolved_basis
            if (
                basis.ndim != 2
                or basis.shape[0] != self.shape[0]
                or basis.device != self.device
                or basis.dtype != self.dtype
                or not torch.isfinite(basis).all()
            ):
                raise ValueError("resolved basis is incompatible with the Fisher")
            gram = basis.mT @ basis
            identity = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
            tolerance = 500 * torch.finfo(basis.dtype).eps * max(1, basis.shape[0])
            if not torch.allclose(gram, identity, rtol=0.0, atol=tolerance):
                raise ValueError("resolved basis must have orthonormal columns")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.base.shape)

    @property
    def device(self) -> torch.device:
        return self.base.device

    @property
    def dtype(self) -> torch.dtype:
        return self.base.dtype

    def _ridge_action(self, vector: Tensor) -> Tensor:
        if self.geometry == "isotropic":
            return vector
        assert self.resolved_basis is not None
        return vector - self.resolved_basis @ (self.resolved_basis.mT @ vector)

    def matvec(self, vector: Tensor) -> Tensor:
        return _matvec(self.base, vector) + self.kappa * self._ridge_action(vector)

    def quadratic(self, vector: Tensor) -> Tensor:
        tail = self._ridge_action(vector)
        return _quadratic(self.base, vector) + self.kappa * torch.sum(vector * tail)

    def diagonal_vector(self) -> Tensor:
        diagonal = _diagonal(self.base)
        if self.geometry == "isotropic":
            return diagonal + self.kappa
        assert self.resolved_basis is not None
        return diagonal + self.kappa * (1 - self.resolved_basis.square().sum(dim=1))

    def to_dense(self) -> Tensor:
        identity = torch.eye(self.shape[0], device=self.device, dtype=self.dtype)
        return self.matvec(identity)

    def damped_solve(self, right_hand_side: Tensor, damping: float) -> Tensor:
        if damping <= 0 or not torch.isfinite(torch.tensor(damping)):
            raise ValueError("damping must be positive and finite")
        identity = torch.eye(self.shape[0], device=self.device, dtype=self.dtype)
        return torch.linalg.solve(self.to_dense() + float(damping) * identity, right_hand_side)

    def inverse_trace(self, damping: float) -> Tensor:
        identity = torch.eye(self.shape[0], device=self.device, dtype=self.dtype)
        return torch.trace(torch.linalg.inv(self.to_dense() + float(damping) * identity))

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "RidgeFisher":
        base = self.base.to(device=device, dtype=dtype)
        basis = None if self.resolved_basis is None else self.resolved_basis.to(device=device, dtype=dtype)
        return RidgeFisher(base, self.kappa, self.geometry, basis)

    def artifact_mapping(self) -> dict[str, Any]:
        return {
            "kind": "plan12_ridge",
            "kappa": self.kappa,
            "geometry": self.geometry,
            "resolved_basis": None if self.resolved_basis is None else self.resolved_basis.detach().cpu(),
        }

    def storage_bytes(self) -> int:
        base_bytes = self.base.numel() * self.base.element_size() if isinstance(self.base, Tensor) else self.base.storage_bytes()
        basis_bytes = 0 if self.resolved_basis is None else self.resolved_basis.numel() * self.resolved_basis.element_size()
        return base_bytes + basis_bytes


def leading_subspace(matrix: Tensor, rank: int) -> tuple[Tensor, Tensor]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not 0 <= rank <= matrix.shape[0]:
        raise ValueError("rank is outside the matrix dimension")
    eigenvalues, eigenvectors = torch.linalg.eigh((matrix + matrix.mT) / 2)
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvalues[order], eigenvectors[:, order[:rank]]


def fisher_scale(fisher: FisherLike) -> float:
    diagonal = _diagonal(fisher)
    return float(diagonal.sum() / diagonal.numel())


def ridge_strengths(
    fisher: FisherLike,
    ratios: tuple[float, ...],
) -> tuple[float, ...]:
    scale = fisher_scale(fisher)
    if not scale > 0:
        raise ValueError("mean Fisher eigenvalue must be positive")
    return tuple(scale * ratio for ratio in ratios)


def displacement_components(vector: Tensor, resolved_basis: Tensor) -> dict[str, float]:
    resolved = resolved_basis.mT @ vector
    unresolved = vector - resolved_basis @ resolved
    return {
        "total_squared": float(vector.square().sum()),
        "resolved_squared": float(resolved.square().sum()),
        "unresolved_squared": float(unresolved.square().sum()),
    }
