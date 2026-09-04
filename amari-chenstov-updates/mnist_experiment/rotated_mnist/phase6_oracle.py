"""Pure numerical pieces of the Plan 6 instantaneous local oracle."""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from src.lanczos_wrapper import LanczosDiagnostics, approximate_low_rank_diagonal
from src.representations import FisherRepresentation, LowRankDiagonalFisher


ANGLE_DECIMALS = 12


def canonical_angle(value: float) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 30.0:
        raise ValueError("oracle angle must be finite and in [0, 30]")
    return round(float(value), ANGLE_DECIMALS)


def angle_key(value: float) -> str:
    scaled = int(round(canonical_angle(value) * 10**ANGLE_DECIMALS))
    return f"angle-{scaled:014d}"


def transition_steps(
    schedules: Mapping[str, Sequence[float]],
    requested: Mapping[str, Sequence[int]],
    *,
    full: bool,
) -> dict[str, tuple[int, ...]]:
    result = {}
    for name, angles in schedules.items():
        maximum = len(angles) - 1
        values = tuple(range(maximum)) if full else tuple(requested[name])
        if any(step < 0 or step >= maximum for step in values):
            raise ValueError(f"{name} oracle transition is out of range")
        result[name] = values
    return result


def required_angles(
    schedules: Mapping[str, Sequence[float]],
    steps: Mapping[str, Sequence[int]],
) -> tuple[float, ...]:
    values = set()
    for name, selected in steps.items():
        angles = schedules[name]
        for step in selected:
            values.add(canonical_angle(float(angles[step])))
            values.add(canonical_angle(float(angles[step + 1])))
    return tuple(sorted(values))


def full_reference_angle_union(
    schedules: Mapping[str, Sequence[float]],
) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                canonical_angle(float(angle))
                for angles in schedules.values()
                for angle in angles
            }
        )
    )


@dataclasses.dataclass(frozen=True)
class ScoreFisherApproximation:
    representation: LowRankDiagonalFisher
    lanczos: LanczosDiagnostics
    sample_count: int
    parameter_count: int
    exact_diagonal: Tensor

    def artifact_mapping(self) -> dict[str, Any]:
        return {
            "representation": self.representation.artifact_mapping(),
            "lanczos": self.lanczos.mapping(),
            "sample_count": self.sample_count,
            "parameter_count": self.parameter_count,
            "exact_diagonal": self.exact_diagonal.detach().cpu(),
        }


def approximate_score_fisher(
    scores: Tensor,
    *,
    rank: int,
    seed: int,
) -> ScoreFisherApproximation:
    if (
        scores.ndim != 2
        or scores.shape[0] < 2
        or scores.shape[1] < 1
        or not scores.is_floating_point()
        or not torch.isfinite(scores).all()
    ):
        raise ValueError("scores must be a finite floating sample-by-parameter matrix")
    matrix = scores.to(dtype=torch.float64)
    count = matrix.shape[0]
    diagonal = matrix.square().mean(dim=0)

    def operator(vector: Tensor) -> Tensor:
        return matrix.mT @ (matrix @ vector) / count

    approximation = approximate_low_rank_diagonal(
        operator,
        diagonal,
        rank=rank,
        seed=seed,
        maximum_krylov_rank=min(rank, count),
    )
    return ScoreFisherApproximation(
        representation=approximation.representation,
        lanczos=approximation.diagnostics,
        sample_count=count,
        parameter_count=matrix.shape[1],
        exact_diagonal=diagonal,
    )


def covariance_risk(
    parameters: Tensor,
    fisher: FisherRepresentation,
    *,
    local_sample_size: int,
) -> tuple[float, Tensor]:
    if (
        parameters.ndim != 2
        or parameters.shape[0] < 2
        or parameters.shape[1] != fisher.shape[0]
        or not parameters.is_floating_point()
        or not torch.isfinite(parameters).all()
    ):
        raise ValueError("local parameters have an invalid shape or value")
    if local_sample_size < 1:
        raise ValueError("local_sample_size must be positive")
    values = parameters.to(device=fisher.device, dtype=fisher.dtype)
    deviations = values - values.mean(dim=0, keepdim=True)
    deviation_columns = deviations.mT.contiguous()
    energies = (deviation_columns * fisher.matvec(deviation_columns)).sum(dim=0)
    risk = local_sample_size * energies.sum() / (values.shape[0] - 1)
    return float(risk), energies.detach().cpu()


def paired_displacement_covariance_risk(
    parameters_from: Tensor,
    parameters_to: Tensor,
    fisher: FisherRepresentation,
    *,
    local_sample_size: int,
) -> tuple[float, Tensor]:
    """Estimate ``tr(I K_delta)`` from paired local-estimator paths.

    Replicate ``b`` must use the same sampled observation identities and
    optimizer randomness at both parameter values. This preserves the
    cross-covariance in ``K_delta = M Cov(theta_hat_to - theta_hat_from)``.
    """

    if parameters_from.shape != parameters_to.shape:
        raise ValueError("paired local parameters must have identical shapes")
    displacements = parameters_to - parameters_from
    return covariance_risk(
        displacements,
        fisher,
        local_sample_size=local_sample_size,
    )


@dataclasses.dataclass(frozen=True)
class OracleEstimate:
    signal_raw: float
    reference_displacement_shape_risk: float
    signal_noise_correction: float
    signal: float
    old_covariance_shape_risk: float
    new_covariance_shape_risk: float
    old_covariance_risk: float
    new_covariance_risk: float
    q: float
    deployed_batch_size: int
    pi: float

    @property
    def curvature(self) -> float:
        return self.signal + self.old_covariance_risk + self.new_covariance_risk

    def mapping(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "risk_curvature": self.curvature}


def estimate_oracle(
    reference_from: Tensor,
    reference_to: Tensor,
    local_from: Tensor,
    local_to: Tensor,
    fisher: FisherRepresentation,
    *,
    reference_sample_size: int,
    local_sample_size: int,
    q: float,
    deployed_batch_size: int,
) -> OracleEstimate:
    if reference_sample_size < 1 or deployed_batch_size < 1:
        raise ValueError("oracle sample sizes must be positive")
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError("q must be finite and positive")
    references = (reference_from, reference_to)
    if any(
        value.ndim != 1
        or value.numel() != fisher.shape[0]
        or not torch.isfinite(value).all()
        for value in references
    ):
        raise ValueError("reference parameters are invalid")
    old_shape, _ = covariance_risk(
        local_from, fisher, local_sample_size=local_sample_size
    )
    new_shape, _ = covariance_risk(
        local_to, fisher, local_sample_size=local_sample_size
    )
    displacement = (reference_to - reference_from).to(
        device=fisher.device, dtype=fisher.dtype
    )
    signal_raw = float(fisher.quadratic(displacement))
    displacement_shape, _ = paired_displacement_covariance_risk(
        local_from,
        local_to,
        fisher,
        local_sample_size=local_sample_size,
    )
    noise_correction = displacement_shape / reference_sample_size
    signal = max(0.0, signal_raw - noise_correction)
    old_covariance = q * old_shape
    new_covariance = new_shape / deployed_batch_size
    denominator = signal + old_covariance + new_covariance
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("oracle risk denominator is not positive and finite")
    pi = (signal + old_covariance) / denominator
    return OracleEstimate(
        signal_raw=signal_raw,
        reference_displacement_shape_risk=displacement_shape,
        signal_noise_correction=noise_correction,
        signal=signal,
        old_covariance_shape_risk=old_shape,
        new_covariance_shape_risk=new_shape,
        old_covariance_risk=old_covariance,
        new_covariance_risk=new_covariance,
        q=q,
        deployed_batch_size=deployed_batch_size,
        pi=pi,
    )


def bootstrap_oracle_pi(
    reference_from: Tensor,
    reference_to: Tensor,
    local_from: Tensor,
    local_to: Tensor,
    fisher: FisherRepresentation,
    *,
    reference_sample_size: int,
    local_sample_size: int,
    q: float,
    deployed_batch_size: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if replicates < 20:
        raise ValueError("bootstrap requires at least 20 replicates")
    if local_from.shape != local_to.shape:
        raise ValueError("bootstrap requires paired local parameter arrays")
    count = local_from.shape[0]
    paired_displacements = local_to - local_from
    centered_displacements = (
        paired_displacements - paired_displacements.mean(dim=0, keepdim=True)
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = []
    for _ in range(replicates):
        indices = torch.randint(count, (count,), generator=generator)
        reference_draw = int(
            torch.randint(count, (1,), generator=generator).item()
        )
        reference_noise = centered_displacements[reference_draw] * math.sqrt(
            local_sample_size / reference_sample_size
        )
        estimate = estimate_oracle(
            reference_from,
            reference_to + reference_noise,
            local_from[indices],
            local_to[indices],
            fisher,
            reference_sample_size=reference_sample_size,
            local_sample_size=local_sample_size,
            q=q,
            deployed_batch_size=deployed_batch_size,
        )
        values.append(estimate.pi)
    ordered = sorted(values)
    lower_index = max(0, int(math.floor(0.025 * (replicates - 1))))
    upper_index = min(replicates - 1, int(math.ceil(0.975 * (replicates - 1))))
    standard_error = statistics.stdev(values)
    return {
        "mean": statistics.fmean(values),
        "standard_error": standard_error,
        "lower_95": ordered[lower_index],
        "upper_95": ordered[upper_index],
    }
