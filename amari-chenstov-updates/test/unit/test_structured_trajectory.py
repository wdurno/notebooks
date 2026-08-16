import torch
import pytest

from src.fisher import LFUBatchEstimate
from src.fixed_trajectory import OnlineStepStatistics
from src.structured_trajectory import (
    _candidate_spectrum_diagnostics,
    DiagonalFisherTracker,
    LowRankDiagonalFisherTracker,
    structured_representation_metrics,
)


def test_candidate_spectrum_tolerates_rank_deficiency() -> None:
    vector = torch.linspace(-2.0, 3.0, 64, dtype=torch.float64)
    candidate = torch.outer(vector, vector)

    diagnostics = _candidate_spectrum_diagnostics(candidate)

    assert diagnostics.materially_negative_eigenvalue_count == 0
    assert diagnostics.numerical_rank == 1
    assert diagnostics.tolerance > 0.0
    assert diagnostics.backend == "cpu"


def test_candidate_spectrum_retries_eigh_nonconvergence_on_cpu_float64(
    monkeypatch,
) -> None:
    original = torch.linalg.eigvalsh
    calls = []

    def flaky_eigvalsh(candidate: torch.Tensor) -> torch.Tensor:
        calls.append((candidate.device.type, candidate.dtype))
        if len(calls) == 1:
            raise RuntimeError("linalg.eigh: The algorithm failed to converge")
        return original(candidate)

    monkeypatch.setattr(torch.linalg, "eigvalsh", flaky_eigvalsh)
    diagnostics = _candidate_spectrum_diagnostics(
        torch.eye(4, dtype=torch.float32)
    )

    assert calls == [("cpu", torch.float32), ("cpu", torch.float64)]
    assert diagnostics.backend == "cpu_float64_fallback"
    assert diagnostics.minimum_eigenvalue == pytest.approx(1.0)


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


def test_tracker_caps_krylov_depth_for_rank_one_candidate() -> None:
    vector = torch.linspace(-2.0, 3.0, 64, dtype=torch.float64)
    candidate = torch.outer(vector, vector)
    tracker = LowRankDiagonalFisherTracker(
        candidate,
        rank=8,
        ema_gain=1.0,
        ridge_half_life_steps=2.0,
        ridge_amplitude_epsilon=1e-6,
        ridge_coherence_threshold=0.75,
        correction_method="ema",
    )

    update = tracker.update(
        0,
        _statistics(
            candidate,
            torch.zeros_like(candidate),
            torch.zeros_like(candidate),
        ),
        torch.zeros(64, dtype=torch.float64),
        lanczos_seed=7,
    )

    assert update.candidate_numerical_rank == 1
    assert update.lanczos.requested_rank == 8
    assert update.lanczos.krylov_rank_limit == 2
    assert update.lanczos.executed_krylov_rank <= 2
    assert update.lanczos.realized_rank == 1
    assert update.lanczos.represented_diagonal_relative_error < 1e-10


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


def test_structured_correction_modes_isolate_ac_and_residual_terms() -> None:
    initial = torch.eye(2, dtype=torch.float64)
    first = _statistics(
        initial,
        torch.zeros_like(initial),
        torch.zeros_like(initial),
    )
    second = _statistics(
        initial,
        torch.diag(torch.tensor([0.4, -0.2], dtype=torch.float64)),
        torch.diag(torch.tensor([0.3, 0.1], dtype=torch.float64)),
    )
    direction_zero = torch.zeros(2, dtype=torch.float64)
    direction = torch.tensor([0.2, -0.1], dtype=torch.float64)

    corrections = {}
    for method in ("ema", "ac_only", "full_lfu"):
        tracker = DiagonalFisherTracker(
            initial,
            ema_gain=0.25,
            ridge_half_life_steps=2.0,
            ridge_amplitude_epsilon=1e-6,
            ridge_coherence_threshold=0.75,
            correction_method=method,
        )
        tracker.update(0, first, direction_zero)
        update = tracker.update(1, second, direction)
        corrections[method] = update.correction
        assert update.correction_method == method

    torch.testing.assert_close(
        corrections["ema"],
        torch.zeros_like(corrections["ema"]),
    )
    assert torch.linalg.vector_norm(corrections["ac_only"]) > 0
    assert not torch.allclose(
        corrections["full_lfu"],
        corrections["ac_only"],
    )


def test_structured_tracker_rejects_unknown_correction_method() -> None:
    with pytest.raises(ValueError, match="unsupported structured correction"):
        DiagonalFisherTracker(
            torch.eye(2, dtype=torch.float64),
            ema_gain=0.25,
            ridge_half_life_steps=2.0,
            ridge_amplitude_epsilon=1e-6,
            ridge_coherence_threshold=0.75,
            correction_method="mystery",
        )
