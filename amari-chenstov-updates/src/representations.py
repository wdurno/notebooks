"""Dense, diagonal, and low-rank-plus-diagonal Fisher representations."""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class FisherRepresentation(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def matvec(self, vector: Tensor) -> Tensor: ...

    def quadratic(self, vector: Tensor) -> Tensor: ...

    def diagonal_vector(self) -> Tensor: ...

    def damped_solve(self, right_hand_side: Tensor, damping: float) -> Tensor: ...

    def inverse_trace(self, damping: float) -> Tensor: ...

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "FisherRepresentation": ...

    def artifact_mapping(self) -> dict[str, Any]: ...

    def storage_bytes(self) -> int: ...


def _validate_vector(
    vector: Tensor,
    *,
    parameter_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if vector.ndim not in {1, 2} or vector.shape[0] != parameter_count:
        raise ValueError("vector must have shape (parameters,) or (parameters, k)")
    if vector.device != device or vector.dtype != dtype:
        raise ValueError("vector and Fisher representation must share dtype/device")
    if not torch.isfinite(vector).all():
        raise ValueError("vector must be finite")


def _validate_damping(damping: float) -> float:
    if not isinstance(damping, (int, float)):
        raise ValueError("damping must be a positive finite scalar")
    value = float(damping)
    if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
        raise ValueError("damping must be a positive finite scalar")
    return value


@dataclasses.dataclass(frozen=True)
class DenseFisher:
    matrix: Tensor

    def __post_init__(self) -> None:
        if (
            self.matrix.ndim != 2
            or self.matrix.shape[0] != self.matrix.shape[1]
            or not self.matrix.is_floating_point()
            or not torch.isfinite(self.matrix).all()
        ):
            raise ValueError("dense Fisher must be a finite floating square matrix")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.matrix.shape)

    @property
    def device(self) -> torch.device:
        return self.matrix.device

    @property
    def dtype(self) -> torch.dtype:
        return self.matrix.dtype

    def matvec(self, vector: Tensor) -> Tensor:
        _validate_vector(
            vector,
            parameter_count=self.shape[0],
            device=self.device,
            dtype=self.dtype,
        )
        return self.matrix @ vector

    def quadratic(self, vector: Tensor) -> Tensor:
        if vector.ndim != 1:
            raise ValueError("quadratic form requires a parameter vector")
        return vector @ self.matvec(vector)

    def diagonal_vector(self) -> Tensor:
        return torch.diagonal(self.matrix)

    def damped_solve(self, right_hand_side: Tensor, damping: float) -> Tensor:
        value = _validate_damping(damping)
        _validate_vector(
            right_hand_side,
            parameter_count=self.shape[0],
            device=self.device,
            dtype=self.dtype,
        )
        identity = torch.eye(
            self.shape[0],
            device=self.device,
            dtype=self.dtype,
        )
        return torch.linalg.solve(
            self.matrix + value * identity,
            right_hand_side,
        )

    def inverse_trace(self, damping: float) -> Tensor:
        value = _validate_damping(damping)
        eigenvalues = torch.linalg.eigvalsh(
            (self.matrix + self.matrix.mT) / 2
        )
        return torch.sum(torch.reciprocal(eigenvalues + value))

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "DenseFisher":
        return DenseFisher(self.matrix.to(device=device, dtype=dtype))

    def artifact_mapping(self) -> dict[str, Any]:
        return {"kind": "dense", "matrix": self.matrix.detach().cpu()}

    def storage_bytes(self) -> int:
        return self.matrix.numel() * self.matrix.element_size()


@dataclasses.dataclass(frozen=True)
class DiagonalFisher:
    values: Tensor

    def __post_init__(self) -> None:
        if (
            self.values.ndim != 1
            or not self.values.is_floating_point()
            or not torch.isfinite(self.values).all()
            or bool((self.values < 0).any())
        ):
            raise ValueError(
                "diagonal Fisher values must be a finite nonnegative vector"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.values.numel(), self.values.numel())

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    def matvec(self, vector: Tensor) -> Tensor:
        _validate_vector(
            vector,
            parameter_count=self.shape[0],
            device=self.device,
            dtype=self.dtype,
        )
        if vector.ndim == 1:
            return self.values * vector
        return self.values.unsqueeze(1) * vector

    def quadratic(self, vector: Tensor) -> Tensor:
        if vector.ndim != 1:
            raise ValueError("quadratic form requires a parameter vector")
        return torch.sum(self.values * vector.square())

    def diagonal_vector(self) -> Tensor:
        return self.values

    def damped_solve(self, right_hand_side: Tensor, damping: float) -> Tensor:
        value = _validate_damping(damping)
        _validate_vector(
            right_hand_side,
            parameter_count=self.shape[0],
            device=self.device,
            dtype=self.dtype,
        )
        denominator = self.values + value
        if right_hand_side.ndim == 1:
            return right_hand_side / denominator
        return right_hand_side / denominator.unsqueeze(1)

    def inverse_trace(self, damping: float) -> Tensor:
        value = _validate_damping(damping)
        return torch.sum(torch.reciprocal(self.values + value))

    def to_dense(self) -> Tensor:
        return torch.diag(self.values)

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "DiagonalFisher":
        return DiagonalFisher(self.values.to(device=device, dtype=dtype))

    def artifact_mapping(self) -> dict[str, Any]:
        return {
            "kind": "diagonal",
            "values": self.values.detach().cpu(),
        }

    def storage_bytes(self) -> int:
        return self.values.numel() * self.values.element_size()


@dataclasses.dataclass(frozen=True)
class LowRankDiagonalFisher:
    factor: Tensor
    residual_diagonal: Tensor

    def __post_init__(self) -> None:
        if (
            self.factor.ndim != 2
            or self.residual_diagonal.ndim != 1
            or self.factor.shape[0] != self.residual_diagonal.numel()
            or not self.factor.is_floating_point()
            or self.factor.dtype != self.residual_diagonal.dtype
            or self.factor.device != self.residual_diagonal.device
            or not torch.isfinite(self.factor).all()
            or not torch.isfinite(self.residual_diagonal).all()
            or bool((self.residual_diagonal < 0).any())
        ):
            raise ValueError(
                "low-rank factor and nonnegative residual diagonal are invalid"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.factor.shape[0], self.factor.shape[0])

    @property
    def rank(self) -> int:
        return self.factor.shape[1]

    @property
    def device(self) -> torch.device:
        return self.factor.device

    @property
    def dtype(self) -> torch.dtype:
        return self.factor.dtype

    def matvec(self, vector: Tensor) -> Tensor:
        _validate_vector(
            vector,
            parameter_count=self.shape[0],
            device=self.device,
            dtype=self.dtype,
        )
        low_rank = self.factor @ (self.factor.mT @ vector)
        if vector.ndim == 1:
            return low_rank + self.residual_diagonal * vector
        return low_rank + self.residual_diagonal.unsqueeze(1) * vector

    def quadratic(self, vector: Tensor) -> Tensor:
        if vector.ndim != 1:
            raise ValueError("quadratic form requires a parameter vector")
        projected = self.factor.mT @ vector
        return projected.square().sum() + torch.sum(
            self.residual_diagonal * vector.square()
        )

    def diagonal_vector(self) -> Tensor:
        return self.factor.square().sum(dim=1) + self.residual_diagonal

    def damped_solve(self, right_hand_side: Tensor, damping: float) -> Tensor:
        value = _validate_damping(damping)
        _validate_vector(
            right_hand_side,
            parameter_count=self.shape[0],
            device=self.device,
            dtype=self.dtype,
        )
        diagonal = self.residual_diagonal + value
        inverse_diagonal = torch.reciprocal(diagonal)
        scaled_rhs = (
            inverse_diagonal * right_hand_side
            if right_hand_side.ndim == 1
            else inverse_diagonal.unsqueeze(1) * right_hand_side
        )
        if self.rank == 0:
            return scaled_rhs
        scaled_factor = inverse_diagonal.unsqueeze(1) * self.factor
        middle = (
            torch.eye(self.rank, device=self.device, dtype=self.dtype)
            + self.factor.mT @ scaled_factor
        )
        correction = scaled_factor @ torch.linalg.solve(
            middle,
            self.factor.mT @ scaled_rhs,
        )
        return scaled_rhs - correction

    def inverse_trace(self, damping: float) -> Tensor:
        value = _validate_damping(damping)
        inverse_diagonal = torch.reciprocal(
            self.residual_diagonal + value
        )
        result = inverse_diagonal.sum()
        if self.rank == 0:
            return result
        scaled_factor = inverse_diagonal.unsqueeze(1) * self.factor
        middle = (
            torch.eye(self.rank, device=self.device, dtype=self.dtype)
            + self.factor.mT @ scaled_factor
        )
        squared_scaled_gram = scaled_factor.mT @ scaled_factor
        return result - torch.trace(
            torch.linalg.solve(middle, squared_scaled_gram)
        )

    def to_dense(self) -> Tensor:
        return self.factor @ self.factor.mT + torch.diag(
            self.residual_diagonal
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LowRankDiagonalFisher":
        return LowRankDiagonalFisher(
            self.factor.to(device=device, dtype=dtype),
            self.residual_diagonal.to(device=device, dtype=dtype),
        )

    def artifact_mapping(self) -> dict[str, Any]:
        return {
            "kind": "low_rank_diagonal",
            "factor": self.factor.detach().cpu(),
            "residual_diagonal": self.residual_diagonal.detach().cpu(),
        }

    def storage_bytes(self) -> int:
        return (
            self.factor.numel() * self.factor.element_size()
            + self.residual_diagonal.numel()
            * self.residual_diagonal.element_size()
        )


def representation_from_artifact(
    value: dict[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> DenseFisher | DiagonalFisher | LowRankDiagonalFisher:
    kind = value.get("kind")
    if kind == "dense":
        return DenseFisher(value["matrix"].to(device=device))
    if kind == "diagonal":
        return DiagonalFisher(value["values"].to(device=device))
    if kind == "low_rank_diagonal":
        return LowRankDiagonalFisher(
            value["factor"].to(device=device),
            value["residual_diagonal"].to(device=device),
        )
    raise ValueError(f"unsupported Fisher representation artifact: {kind}")


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
