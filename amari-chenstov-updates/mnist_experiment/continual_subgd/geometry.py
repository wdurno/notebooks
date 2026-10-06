"""Low-rank adaptation covariance and Plan 13 preconditioners."""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import Tensor


def half_life_gain(half_life: float) -> float:
    value = float(half_life)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("half-life must be positive and finite")
    return 1.0 - 2.0 ** (-1.0 / value)


def _canonicalize_columns(basis: Tensor) -> Tensor:
    if basis.numel() == 0:
        return basis
    indices = torch.argmax(torch.abs(basis), dim=0)
    signs = torch.sign(basis[indices, torch.arange(basis.shape[1], device=basis.device)])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return basis * signs


@dataclasses.dataclass(frozen=True)
class AdaptationGeometry:
    basis: Tensor
    eigenvalues: Tensor

    def __post_init__(self) -> None:
        if self.basis.ndim != 2 or self.eigenvalues.ndim != 1:
            raise ValueError("basis and eigenvalues must be a matrix and vector")
        if self.basis.shape[1] != self.eigenvalues.numel() or self.basis.shape[1] < 1:
            raise ValueError("basis rank and eigenvalue count differ")
        if self.basis.device != self.eigenvalues.device or self.basis.dtype != self.eigenvalues.dtype:
            raise ValueError("basis and eigenvalues must share dtype and device")
        if not torch.isfinite(self.basis).all() or not torch.isfinite(self.eigenvalues).all():
            raise ValueError("geometry must be finite")
        if torch.any(self.eigenvalues < 0):
            raise ValueError("adaptation eigenvalues must be nonnegative")
        gram = self.basis.mT @ self.basis
        identity = torch.eye(self.rank, dtype=gram.dtype, device=gram.device)
        tolerance = 256 * torch.finfo(gram.dtype).eps * max(self.parameter_count, 1)
        if not torch.allclose(gram, identity, atol=tolerance, rtol=tolerance):
            raise ValueError("adaptation basis must be orthonormal")

    @property
    def parameter_count(self) -> int:
        return self.basis.shape[0]

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @classmethod
    def from_observations(cls, observations: Tensor, rank: int) -> "AdaptationGeometry":
        if observations.ndim != 2 or observations.shape[0] < 1:
            raise ValueError("observations must have shape (count, parameters)")
        if rank < 1 or rank > min(observations.shape):
            raise ValueError("rank exceeds the observation matrix")
        if not torch.isfinite(observations).all():
            raise ValueError("observations must be finite")
        _, singular, right = torch.linalg.svd(observations, full_matrices=False)
        basis = _canonicalize_columns(right[:rank].mT.contiguous())
        eigenvalues = singular[:rank].square() / observations.shape[0]
        return cls(basis, eigenvalues)

    def normalized_eigenvalues(self, epsilon: float = 1e-12) -> Tensor:
        denominator = self.eigenvalues.sum().clamp_min(float(epsilon))
        return self.rank * self.eigenvalues / denominator

    def project(self, vector: Tensor) -> Tensor:
        self._validate_vector(vector)
        return self.basis @ (self.basis.mT @ vector)

    def innovation(self, vector: Tensor, epsilon: float = 1e-12) -> float:
        self._validate_vector(vector)
        residual = vector - self.project(vector)
        return float(residual.square().sum() / (vector.square().sum() + float(epsilon)))

    def precondition(
        self,
        vector: Tensor,
        *,
        alpha: float = 1.0,
        epsilon: float = 0.0,
        projector_only: bool = False,
    ) -> Tensor:
        self._validate_vector(vector)
        alpha_value = float(alpha)
        epsilon_value = float(epsilon)
        if not 0 <= alpha_value <= 1 or epsilon_value < 0:
            raise ValueError("alpha must lie in [0, 1] and epsilon must be nonnegative")
        coefficients = self.basis.mT @ vector
        parallel = self.basis @ coefficients
        if projector_only:
            learned = parallel
        else:
            learned = self.basis @ (self.normalized_eigenvalues() * coefficients)
        orthogonal = vector - parallel
        return (1 - alpha_value) * vector + alpha_value * (
            learned + epsilon_value * orthogonal
        )

    def update(self, observation: Tensor, beta: float) -> "AdaptationGeometry":
        self._validate_vector(observation)
        beta_value = float(beta)
        if not 0 < beta_value <= 1:
            raise ValueError("beta must lie in (0, 1]")
        coordinates = self.basis.mT @ observation
        residual = observation - self.basis @ coordinates
        residual_norm = torch.linalg.vector_norm(residual)
        threshold = (
            64
            * torch.finfo(observation.dtype).eps
            * max(float(torch.linalg.vector_norm(observation)), 1.0)
        )
        diagonal = torch.diag((1 - beta_value) * self.eigenvalues)
        if float(residual_norm) <= threshold:
            small = diagonal + beta_value * torch.outer(coordinates, coordinates)
            values, rotation = torch.linalg.eigh((small + small.mT) / 2)
            order = torch.argsort(values, descending=True)
            basis = self.basis @ rotation[:, order]
            values = values[order]
        else:
            direction = residual / residual_norm
            augmented = torch.cat((coordinates, residual_norm.reshape(1)))
            small = torch.zeros(
                self.rank + 1,
                self.rank + 1,
                dtype=observation.dtype,
                device=observation.device,
            )
            small[: self.rank, : self.rank] = diagonal
            small += beta_value * torch.outer(augmented, augmented)
            values, rotation = torch.linalg.eigh((small + small.mT) / 2)
            order = torch.argsort(values, descending=True)[: self.rank]
            basis = torch.cat((self.basis, direction[:, None]), dim=1) @ rotation[:, order]
            values = values[order]
        basis, triangular = torch.linalg.qr(basis, mode="reduced")
        correction = triangular @ torch.diag(values) @ triangular.mT
        values, rotation = torch.linalg.eigh((correction + correction.mT) / 2)
        order = torch.argsort(values, descending=True)
        basis = _canonicalize_columns(basis @ rotation[:, order])
        return AdaptationGeometry(basis, values[order].clamp_min(0))

    def update_growing(
        self,
        observation: Tensor,
        beta: float,
        *,
        rank_cap: int,
    ) -> "AdaptationGeometry":
        """Apply an exponential rank-one update, retaining a new direction when available."""

        self._validate_vector(observation)
        beta_value = float(beta)
        if not 0 < beta_value <= 1:
            raise ValueError("beta must lie in (0, 1]")
        if not self.rank <= rank_cap <= self.parameter_count:
            raise ValueError("rank cap must lie between current rank and parameter count")
        coordinates = self.basis.mT @ observation
        residual = observation - self.basis @ coordinates
        residual_norm = torch.linalg.vector_norm(residual)
        threshold = (
            64
            * torch.finfo(observation.dtype).eps
            * max(float(torch.linalg.vector_norm(observation)), 1.0)
        )
        if float(residual_norm) <= threshold or self.rank == rank_cap:
            return self.update(observation, beta_value)

        direction = residual / residual_norm
        augmented = torch.cat((coordinates, residual_norm.reshape(1)))
        small = torch.zeros(
            self.rank + 1,
            self.rank + 1,
            dtype=observation.dtype,
            device=observation.device,
        )
        small[: self.rank, : self.rank] = torch.diag(
            (1 - beta_value) * self.eigenvalues
        )
        small += beta_value * torch.outer(augmented, augmented)
        values, rotation = torch.linalg.eigh((small + small.mT) / 2)
        order = torch.argsort(values, descending=True)
        basis = torch.cat((self.basis, direction[:, None]), dim=1) @ rotation[:, order]
        values = values[order]
        basis, triangular = torch.linalg.qr(basis, mode="reduced")
        correction = triangular @ torch.diag(values) @ triangular.mT
        values, rotation = torch.linalg.eigh((correction + correction.mT) / 2)
        order = torch.argsort(values, descending=True)
        basis = _canonicalize_columns(basis @ rotation[:, order])
        return AdaptationGeometry(basis, values[order].clamp_min(0))

    def dense(self) -> Tensor:
        return (self.basis * self.eigenvalues) @ self.basis.mT

    def mapping(self) -> dict[str, object]:
        return {
            "parameter_count": self.parameter_count,
            "rank": self.rank,
            "eigenvalues": [float(value) for value in self.eigenvalues],
        }

    def _validate_vector(self, vector: Tensor) -> None:
        if vector.shape != (self.parameter_count,):
            raise ValueError("vector has the wrong parameter dimension")
        if vector.device != self.basis.device or vector.dtype != self.basis.dtype:
            raise ValueError("vector and geometry must share dtype and device")
        if not torch.isfinite(vector).all():
            raise ValueError("vector must be finite")


def random_geometry(
    parameter_count: int,
    eigenvalues: Tensor,
    *,
    seed: int,
) -> AdaptationGeometry:
    if parameter_count < eigenvalues.numel():
        raise ValueError("random geometry rank exceeds parameter count")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    matrix = torch.randn(
        parameter_count,
        eigenvalues.numel(),
        generator=generator,
        dtype=torch.float64,
    )
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    basis = _canonicalize_columns(basis).to(
        device=eigenvalues.device,
        dtype=eigenvalues.dtype,
    )
    return AdaptationGeometry(basis, eigenvalues.detach().clone())


def projector_distance(left: Tensor, right: Tensor) -> float:
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("bases must have the same matrix shape")
    difference = left @ left.mT - right @ right.mT
    return float(torch.linalg.matrix_norm(difference, ord="fro") / math.sqrt(2.0))
