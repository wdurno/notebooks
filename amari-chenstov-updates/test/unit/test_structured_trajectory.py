import torch

from src.fisher import LFUBatchEstimate
from src.fixed_trajectory import OnlineStepStatistics
from src.structured_trajectory import (
    DiagonalFisherTracker,
    LowRankDiagonalFisherTracker,
    structured_representation_metrics,
)


def _statistics(
    fisher: torch.Tensor,
    amari_chentsov: torch.Tensor,
    residual: torch.Tensor,
) -> OnlineStepStatistics:
    return OnlineStepStatistics(
        estimate=LFUBatchEstimate(fisher, amari_chentsov, residual),
        score_gradient_count=4,
        hvp_count=4,
        elapsed_seconds=0.0,
    )


def _trackers(
    initial: torch.Tensor,
) -> tuple[DiagonalFisherTracker, LowRankDiagonalFisherTracker]:
    arguments = {
        "initial_fisher": initial,
        "ema_gain": 0.25,
        "ridge_half_life_steps": 2.0,
        "ridge_amplitude_epsilon": 1e-6,
        "ridge_coherence_threshold": 0.75,
    }
    return (
        DiagonalFisherTracker(**arguments),
        LowRankDiagonalFisherTracker(**arguments, rank=0),
    )


def test_rank_zero_tracker_matches_explicit_diagonal_tracker() -> None:
    initial = torch.tensor(
        [[2.0, 0.4, 0.0], [0.4, 1.0, 0.2], [0.0, 0.2, 0.5]],
        dtype=torch.float64,
    )
    diagonal, rank_zero = _trackers(initial)
    direction_zero = torch.zeros(3, dtype=torch.float64)
    first = _statistics(
        initial,
        torch.zeros_like(initial),
        torch.zeros_like(initial),
    )
    direction = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float64)
    direct = torch.tensor(
        [[2.2, 0.3, 0.1], [0.3, 0.8, 0.0], [0.1, 0.0, 0.7]],
        dtype=torch.float64,
    )
    amari_chentsov = torch.diag(
        torch.tensor([0.1, -0.2, 0.05], dtype=torch.float64)
    )
    residual = torch.tensor(
        [[0.02, 0.01, 0.0], [0.01, 0.03, -0.01], [0.0, -0.01, -0.02]],
        dtype=torch.float64,
    )
    second = _statistics(direct, amari_chentsov, residual)

    diagonal_first = diagonal.update(0, first, direction_zero)
    rank_zero_first = rank_zero.update(
        0,
        first,
        direction_zero,
        lanczos_seed=1,
    )
    diagonal_second = diagonal.update(1, second, direction)
    rank_zero_second = rank_zero.update(
        1,
        second,
        direction,
        lanczos_seed=2,
    )

    torch.testing.assert_close(
        rank_zero_first.representation.residual_diagonal,
        diagonal_first.representation.values,
    )
    torch.testing.assert_close(
        rank_zero_second.representation.residual_diagonal,
        diagonal_second.representation.values,
    )
    assert rank_zero_second.lanczos.realized_rank == 0


def test_structured_metrics_are_zero_for_exact_target() -> None:
    initial = torch.diag(
        torch.tensor([2.0, 1.0, 0.5], dtype=torch.float64)
    )
    diagonal, _ = _trackers(initial)
    update = diagonal.update(
        0,
        _statistics(
            initial,
            torch.zeros_like(initial),
            torch.zeros_like(initial),
        ),
        torch.zeros(3, dtype=torch.float64),
    )
    probes = torch.eye(3, dtype=torch.float64)
    direction = torch.tensor([1.0, 0.5, -0.25], dtype=torch.float64)

    metrics = structured_representation_metrics(
        update.representation,
        initial,
        initial,
        probes,
        direction,
    )

    assert metrics["relative_frobenius_error_to_dense"] == 0.0
    assert metrics["relative_frobenius_error_to_reference"] == 0.0
    assert metrics["fixed_probe_matvec_relative_error"] == 0.0
    assert metrics["mean_probe_quadratic_relative_error"] == 0.0
    assert metrics["directional_quadratic_relative_error"] == 0.0
    assert metrics["leading_eigenvalue_relative_error"] == 0.0
    assert metrics["leading_eigenvector_alignment"] == 1.0
