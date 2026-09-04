"""Pure diagnostics for the Plan 5 learnability and geometry audit."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from .audit_config import GateConfig


def _finite_matrix(matrix: Tensor, name: str) -> Tensor:
    value = matrix.detach().to(device="cpu", dtype=torch.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError(f"{name} must be square")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return 0.5 * (value + value.mT)


def _finite_vector(vector: Tensor, size: int, name: str) -> Tensor:
    value = vector.detach().to(device="cpu", dtype=torch.float64)
    if value.ndim != 1 or value.numel() != size:
        raise ValueError(f"{name} must have shape ({size},)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def fisher_summary(matrix: Tensor) -> dict[str, float]:
    value = _finite_matrix(matrix, "Fisher")
    eigenvalues = torch.linalg.eigvalsh(value)
    trace = float(torch.trace(value))
    frobenius = float(torch.linalg.matrix_norm(value, ord="fro"))
    square_sum = frobenius**2
    return {
        "trace": trace,
        "frobenius_norm": frobenius,
        "spectral_norm": float(eigenvalues[-1]),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "effective_rank_trace_squared": (
            0.0 if square_sum == 0.0 else trace**2 / square_sum
        ),
        "negative_eigenvalue_count": int((eigenvalues < -1e-10).sum()),
    }


def reference_path_diagnostics(
    angles: Sequence[float],
    parameters: Sequence[Tensor],
    fishers: Sequence[Tensor],
) -> list[dict[str, float]]:
    if not (len(angles) == len(parameters) == len(fishers)) or len(angles) < 2:
        raise ValueError("reference path inputs must have equal length >= 2")
    size = fishers[0].shape[0]
    values = [_finite_matrix(matrix, "Fisher") for matrix in fishers]
    vectors = [_finite_vector(vector, size, "parameters") for vector in parameters]
    rows = []
    for index in range(len(angles) - 1):
        left = values[index]
        right = values[index + 1]
        displacement = vectors[index + 1] - vectors[index]
        euclidean_squared = float(displacement @ displacement)
        left_quadratic = max(float(displacement @ (left @ displacement)), 0.0)
        right_quadratic = max(float(displacement @ (right @ displacement)), 0.0)
        fisher_change = float(torch.linalg.matrix_norm(right - left, ord="fro"))
        left_norm = float(torch.linalg.matrix_norm(left, ord="fro"))
        denominator = left_quadratic + right_quadratic
        rows.append(
            {
                "from_angle_degrees": float(angles[index]),
                "to_angle_degrees": float(angles[index + 1]),
                "angle_increment_degrees": float(angles[index + 1] - angles[index]),
                "euclidean_displacement": math.sqrt(euclidean_squared),
                "euclidean_displacement_squared": euclidean_squared,
                "left_fisher_quadratic": left_quadratic,
                "right_fisher_quadratic": right_quadratic,
                "mean_fisher_quadratic": 0.5 * (left_quadratic + right_quadratic),
                "fisher_directional_asymmetry": (
                    0.0
                    if denominator == 0.0
                    else abs(right_quadratic - left_quadratic) / denominator
                ),
                "fisher_frobenius_change": fisher_change,
                "relative_fisher_frobenius_change": fisher_change
                / max(left_norm, torch.finfo(torch.float64).eps),
            }
        )
    return rows


def repeated_fit_noise(
    endpoint_fits: Mapping[float, Sequence[tuple[str, Tensor, Tensor]]],
) -> list[dict[str, Any]]:
    """Compare each endpoint repeat with its primary fit in the primary metric."""

    rows: list[dict[str, Any]] = []
    for angle, fits in sorted(endpoint_fits.items()):
        if len(fits) < 2:
            raise ValueError("each endpoint needs a primary fit and at least one repeat")
        primary_id, primary_parameter, primary_fisher = fits[0]
        fisher = _finite_matrix(primary_fisher, "primary Fisher")
        parameter = _finite_vector(primary_parameter, fisher.shape[0], "parameters")
        fisher_norm = float(torch.linalg.matrix_norm(fisher, ord="fro"))
        for fit_id, repeat_parameter, repeat_fisher in fits[1:]:
            other_fisher = _finite_matrix(repeat_fisher, "repeat Fisher")
            other_parameter = _finite_vector(
                repeat_parameter, fisher.shape[0], "repeat parameters"
            )
            displacement = other_parameter - parameter
            fisher_change = float(
                torch.linalg.matrix_norm(other_fisher - fisher, ord="fro")
            )
            rows.append(
                {
                    "angle_degrees": float(angle),
                    "primary_fit_id": primary_id,
                    "repeat_fit_id": fit_id,
                    "euclidean_displacement": float(
                        torch.linalg.vector_norm(displacement)
                    ),
                    "primary_fisher_quadratic": max(
                        float(displacement @ (fisher @ displacement)), 0.0
                    ),
                    "fisher_frobenius_change": fisher_change,
                    "relative_fisher_frobenius_change": fisher_change
                    / max(fisher_norm, torch.finfo(torch.float64).eps),
                }
            )
    return rows


def offline_risk_coefficients(
    path_rows: Sequence[Mapping[str, float]],
    *,
    parameter_count: int,
    initialization_sample_size: int,
    batch_size: int,
    fixed_pi: float,
) -> list[dict[str, float]]:
    """Ideal-covariance local coefficients; this does not actuate a learner."""

    if min(parameter_count, initialization_sample_size, batch_size) < 1:
        raise ValueError("risk dimensions and sample sizes must be positive")
    if not 0.0 < fixed_pi < 1.0:
        raise ValueError("fixed_pi must lie in (0, 1)")
    q = 1.0 / initialization_sample_size
    rows = []
    for row in path_rows:
        signal = float(row["left_fisher_quadratic"])
        old_covariance = q * parameter_count
        new_covariance = parameter_count / batch_size
        old_risk = signal + old_covariance
        new_risk = new_covariance
        recommendation = old_risk / (old_risk + new_risk)
        rows.append(
            {
                "from_angle_degrees": float(row["from_angle_degrees"]),
                "to_angle_degrees": float(row["to_angle_degrees"]),
                "signal_energy": signal,
                "q_before": q,
                "idealized_old_covariance_risk": old_covariance,
                "idealized_new_covariance_risk": new_covariance,
                "instantaneous_unclipped_pi": recommendation,
            }
        )
        q = (1.0 - fixed_pi) ** 2 * q + fixed_pi**2 / batch_size
    return rows


def feasibility_gate(
    *,
    zero_shot_environment_accuracy_30: float,
    reference_environment_accuracy_30: float,
    path_rows: Sequence[Mapping[str, float]],
    noise_rows: Sequence[Mapping[str, float]],
    config: GateConfig,
) -> dict[str, Any]:
    if not path_rows or not noise_rows:
        raise ValueError("feasibility gate requires path and repeat-noise rows")
    path_changes = [float(row["relative_fisher_frobenius_change"]) for row in path_rows]
    noise_changes = [float(row["relative_fisher_frobenius_change"]) for row in noise_rows]
    fisher_snr = statistics.median(path_changes) / max(
        statistics.median(noise_changes), torch.finfo(torch.float64).eps
    )
    learning_room = (
        reference_environment_accuracy_30 - zero_shot_environment_accuracy_30
    )
    checks = {
        "high_data_learnable": reference_environment_accuracy_30
        >= config.minimum_environment_accuracy_30,
        "zero_shot_leaves_learning_room": learning_room
        >= config.minimum_learning_room,
        "fisher_path_exceeds_repeat_noise": fisher_snr
        >= config.minimum_fisher_signal_to_noise,
    }
    return {
        "recommendation": "proceed" if all(checks.values()) else "check_in",
        "checks": checks,
        "zero_shot_environment_accuracy_30": zero_shot_environment_accuracy_30,
        "reference_environment_accuracy_30": reference_environment_accuracy_30,
        "learning_room": learning_room,
        "median_path_relative_fisher_change": statistics.median(path_changes),
        "median_repeat_relative_fisher_change": statistics.median(noise_changes),
        "fisher_change_signal_to_noise": fisher_snr,
        "thresholds": {
            "minimum_environment_accuracy_30": config.minimum_environment_accuracy_30,
            "minimum_learning_room": config.minimum_learning_room,
            "minimum_fisher_signal_to_noise": config.minimum_fisher_signal_to_noise,
        },
    }
