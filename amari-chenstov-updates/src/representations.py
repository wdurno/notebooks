"""Dense Fisher representation operations and projection diagnostics."""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor


@dataclasses.dataclass(frozen=True)
class PSDProjectionDiagnostics:
    symmetry_error_fro: float
    pre_projection_fro: float
    minimum_eigenvalue: float
    negative_eigenvalue_count: int
    negative_spectral_mass: float
    projection_distance_fro: float
    relative_projection_distance: float


@dataclasses.dataclass(frozen=True)
class PSDProjectionResult:
    projected: Tensor
    symmetrized: Tensor
    diagnostics: PSDProjectionDiagnostics


def project_psd_frobenius(matrix: Tensor) -> PSDProjectionResult:
    """Project a real square matrix onto the symmetric PSD cone in Frobenius norm."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not matrix.is_floating_point():
        raise ValueError("matrix must have a floating-point dtype")
    if not torch.isfinite(matrix).all():
        raise ValueError("matrix must contain only finite values")

    transpose = matrix.mT
    symmetrized = (matrix + transpose) / 2
    symmetry_error = torch.linalg.matrix_norm(matrix - transpose, ord="fro")
    pre_projection_fro = torch.linalg.matrix_norm(symmetrized, ord="fro")

    eigenvalues, eigenvectors = torch.linalg.eigh(symmetrized)
    negative = eigenvalues < 0
    clipped = eigenvalues.clamp_min(0)
    projected = (eigenvectors * clipped.unsqueeze(0)) @ eigenvectors.mT
    projected = (projected + projected.mT) / 2

    projection_distance = torch.linalg.matrix_norm(
        projected - symmetrized,
        ord="fro",
    )
    denominator = max(float(pre_projection_fro), torch.finfo(matrix.dtype).eps)
    diagnostics = PSDProjectionDiagnostics(
        symmetry_error_fro=float(symmetry_error),
        pre_projection_fro=float(pre_projection_fro),
        minimum_eigenvalue=float(eigenvalues.min()),
        negative_eigenvalue_count=int(negative.sum()),
        negative_spectral_mass=float((-eigenvalues[negative]).sum()),
        projection_distance_fro=float(projection_distance),
        relative_projection_distance=float(projection_distance) / denominator,
    )
    return PSDProjectionResult(
        projected=projected,
        symmetrized=symmetrized,
        diagnostics=diagnostics,
    )
