import pytest
import torch
from torch import nn

from src.derivatives import per_sample_derivatives
from src.fisher import (
    apply_amari_chentsov,
    apply_empirical_fisher,
    apply_full_lfu,
    apply_residual,
    dense_lfu_estimate,
    sample_lfu_factorization,
)
from src.parameters import ParameterLayout


class GaussianMeanModel(nn.Module):
    def __init__(self, *, squared: bool) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        self.squared = squared

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = self.theta.square() if self.squared else self.theta
        return mean.expand(inputs.shape[0])


def gaussian_nll(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 0.5 * (targets - predictions).square().sum()


def _estimate(squared: bool):
    model = GaussianMeanModel(squared=squared)
    layout = ParameterLayout.from_module(model)
    inputs = torch.zeros(2, 1, dtype=torch.float64)
    mean = model.theta.detach().square() if squared else model.theta.detach()
    targets = mean + torch.tensor([-1.0, 1.0], dtype=torch.float64)
    direction = torch.ones(1, dtype=torch.float64)
    derivatives = per_sample_derivatives(
        model,
        inputs,
        targets,
        gaussian_nll,
        layout,
        direction=direction,
        strategy="loop",
    )
    return derivatives, direction, dense_lfu_estimate(
        derivatives.gradients,
        derivatives.hvps,
        direction,
    )


def test_canonical_gaussian_has_zero_expected_lfu_terms() -> None:
    _, _, estimate = _estimate(squared=False)

    expected_zero = estimate.fisher.new_zeros(1, 1)
    torch.testing.assert_close(estimate.fisher, estimate.fisher.new_ones(1, 1))
    torch.testing.assert_close(estimate.amari_chentsov, expected_zero)
    torch.testing.assert_close(estimate.residual, expected_zero)
    torch.testing.assert_close(estimate.full, expected_zero)


def test_noncanonical_gaussian_residual_recovers_fisher_derivative() -> None:
    _, _, estimate = _estimate(squared=True)

    expected_zero = estimate.fisher.new_zeros(1, 1)
    torch.testing.assert_close(estimate.fisher, estimate.fisher.new_tensor([[4.0]]))
    torch.testing.assert_close(estimate.amari_chentsov, expected_zero)
    torch.testing.assert_close(
        estimate.residual,
        estimate.fisher.new_tensor([[8.0]]),
    )
    torch.testing.assert_close(estimate.full, estimate.fisher.new_tensor([[8.0]]))


def test_sample_ubu_factorization_equals_explicit_lfu() -> None:
    gradient = torch.tensor([1.5, -0.5, 2.0], dtype=torch.float64)
    hvp = torch.tensor([-2.0, 0.25, 1.0], dtype=torch.float64)
    direction = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)

    basis, core = sample_lfu_factorization(gradient, hvp, direction)
    explicit = (
        -(direction @ gradient) * torch.outer(gradient, gradient)
        + torch.outer(hvp, gradient)
        + torch.outer(gradient, hvp)
    )

    assert basis.shape == (3, 2)
    assert core.shape == (2, 2)
    torch.testing.assert_close(basis @ core @ basis.mT, explicit)


def test_batch_estimate_equals_mean_of_sample_estimates() -> None:
    derivatives, direction, batch = _estimate(squared=True)
    samples = [
        dense_lfu_estimate(
            derivatives.gradients[index : index + 1],
            derivatives.hvps[index : index + 1],
            direction,
        )
        for index in range(derivatives.gradients.shape[0])
    ]

    for name in ("fisher", "amari_chentsov", "residual", "full"):
        expected = torch.stack([getattr(sample, name) for sample in samples]).mean(0)
        torch.testing.assert_close(getattr(batch, name), expected)


@pytest.mark.parametrize("probe_columns", [None, 3])
def test_matrix_free_products_equal_dense_products(
    probe_columns: int | None,
) -> None:
    generator = torch.Generator().manual_seed(123)
    gradients = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    hvps = torch.randn(5, 4, generator=generator, dtype=torch.float64)
    direction = torch.randn(4, generator=generator, dtype=torch.float64)
    if probe_columns is None:
        probes = torch.randn(4, generator=generator, dtype=torch.float64)
    else:
        probes = torch.randn(
            4,
            probe_columns,
            generator=generator,
            dtype=torch.float64,
        )
    dense = dense_lfu_estimate(gradients, hvps, direction)

    torch.testing.assert_close(
        apply_empirical_fisher(gradients, probes),
        dense.fisher @ probes,
    )
    torch.testing.assert_close(
        apply_amari_chentsov(gradients, direction, probes),
        dense.amari_chentsov @ probes,
    )
    torch.testing.assert_close(
        apply_residual(gradients, hvps, probes),
        dense.residual @ probes,
    )
    torch.testing.assert_close(
        apply_full_lfu(gradients, hvps, direction, probes),
        dense.full @ probes,
    )
