import pytest
import torch

from src.convergence import HilbertMean


def test_hilbert_mean_uses_sum_of_coordinate_variances() -> None:
    moments = HilbertMean()
    moments.update(torch.tensor([1.0, 2.0], dtype=torch.float64))
    moments.update(torch.tensor([3.0, 6.0], dtype=torch.float64))

    diagnostics = moments.diagnostics(
        sigma=2.0,
        relative_epsilon=0.01,
        absolute_epsilon=1e-8,
        minimum_count=2,
        maximum_count=8,
    )

    torch.testing.assert_close(
        moments.mean,
        torch.tensor([2.0, 4.0], dtype=torch.float64),
    )
    assert moments.empirical_variance == pytest.approx(10.0)
    assert diagnostics["confidence_radius"] == pytest.approx(2.0 * 5.0**0.5)
    assert diagnostics["converged"] is False


def test_absolute_floor_handles_a_zero_mean() -> None:
    moments = HilbertMean()
    for _ in range(8):
        moments.update(torch.zeros(3, dtype=torch.float64))

    diagnostics = moments.diagnostics(
        sigma=6.0,
        relative_epsilon=0.01,
        absolute_epsilon=1e-8,
        minimum_count=8,
        maximum_count=32,
    )

    assert diagnostics["confidence_radius"] == 0.0
    assert diagnostics["stopping_threshold"] == 1e-8
    assert diagnostics["converged"] is True


def test_lag_one_correlation_uses_the_final_mean() -> None:
    moments = HilbertMean()
    for value in (1.0, 2.0, 3.0):
        moments.update(torch.tensor([value], dtype=torch.float64))

    assert moments.lag_one_correlation == pytest.approx(0.0, abs=1e-15)
