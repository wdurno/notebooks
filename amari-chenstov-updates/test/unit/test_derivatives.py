import pytest
import torch
from torch import nn

from src.derivatives import per_sample_derivatives
from src.parameters import ParameterLayout


class GaussianMeanModel(nn.Module):
    def __init__(self, theta: float, *, squared: bool = False) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([theta], dtype=torch.float64))
        self.squared = squared

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = self.theta.square() if self.squared else self.theta
        return mean.expand(inputs.shape[0])


def gaussian_nll(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 0.5 * (targets - predictions).square().sum()


def _toy_batch(
    model: GaussianMeanModel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.zeros(2, 1, dtype=model.theta.dtype, device=model.theta.device)
    mean = model.theta.detach().square() if model.squared else model.theta.detach()
    targets = mean + torch.tensor(
        [-1.0, 1.0],
        dtype=model.theta.dtype,
        device=model.theta.device,
    )
    direction = torch.ones(1, dtype=model.theta.dtype, device=model.theta.device)
    return inputs, targets, direction


@pytest.mark.parametrize("squared", [False, True])
def test_loop_and_vmap_per_sample_derivatives_agree(squared: bool) -> None:
    model = GaussianMeanModel(1.0, squared=squared)
    layout = ParameterLayout.from_module(model)
    inputs, targets, direction = _toy_batch(model)

    loop = per_sample_derivatives(
        model,
        inputs,
        targets,
        gaussian_nll,
        layout,
        direction=direction,
        strategy="loop",
    )
    vectorized = per_sample_derivatives(
        model,
        inputs,
        targets,
        gaussian_nll,
        layout,
        direction=direction,
        strategy="vmap",
    )

    torch.testing.assert_close(vectorized.gradients, loop.gradients)
    torch.testing.assert_close(vectorized.hvps, loop.hvps)
    assert not loop.gradients.requires_grad
    assert not loop.hvps.requires_grad
    assert not vectorized.gradients.requires_grad
    assert not vectorized.hvps.requires_grad
    assert model.theta.grad is None


def test_loss_gradient_is_negative_gaussian_score() -> None:
    model = GaussianMeanModel(0.25)
    layout = ParameterLayout.from_module(model)
    inputs = torch.zeros(3, 1, dtype=torch.float64)
    targets = torch.tensor([-0.5, 0.0, 1.0], dtype=torch.float64)

    derivatives = per_sample_derivatives(
        model,
        inputs,
        targets,
        gaussian_nll,
        layout,
        strategy="loop",
    )
    analytical_scores = targets - model.theta.detach()

    torch.testing.assert_close(
        derivatives.gradients[:, 0],
        -analytical_scores,
    )


def test_singleton_vmap_preserves_sample_dimension() -> None:
    model = GaussianMeanModel(1.0, squared=True)
    layout = ParameterLayout.from_module(model)
    inputs = torch.zeros(1, 1, dtype=torch.float64)
    targets = torch.tensor([2.0], dtype=torch.float64)
    direction = torch.ones(1, dtype=torch.float64)

    derivatives = per_sample_derivatives(
        model,
        inputs,
        targets,
        gaussian_nll,
        layout,
        direction=direction,
        strategy="vmap",
    )

    assert derivatives.gradients.shape == (1, 1)
    assert derivatives.hvps.shape == (1, 1)
    assert torch.isfinite(derivatives.gradients).all()
    assert torch.isfinite(derivatives.hvps).all()


@pytest.mark.parametrize("squared", [False, True])
def test_hvp_agrees_with_central_gradient_difference(squared: bool) -> None:
    model = GaussianMeanModel(1.0, squared=squared)
    layout = ParameterLayout.from_module(model)
    inputs, targets, direction = _toy_batch(model)
    analytical = per_sample_derivatives(
        model,
        inputs,
        targets,
        gaussian_nll,
        layout,
        direction=direction,
        strategy="loop",
    ).hvps
    original = model.theta.detach().clone()

    errors = []
    for epsilon in (1e-3, 1e-4, 1e-5):
        with torch.no_grad():
            model.theta.copy_(original + epsilon * direction)
        plus = per_sample_derivatives(
            model,
            inputs,
            targets,
            gaussian_nll,
            layout,
            strategy="loop",
        ).gradients
        with torch.no_grad():
            model.theta.copy_(original - epsilon * direction)
        minus = per_sample_derivatives(
            model,
            inputs,
            targets,
            gaussian_nll,
            layout,
            strategy="loop",
        ).gradients
        finite_difference = (plus - minus) / (2 * epsilon)
        errors.append(float(torch.linalg.vector_norm(finite_difference - analytical)))

    with torch.no_grad():
        model.theta.copy_(original)

    assert max(errors) < 4e-6
    assert min(errors) < 1e-8
    if squared:
        assert errors[2] < errors[1] < errors[0]
    assert model.theta.grad is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_gpu_derivatives_agree() -> None:
    cpu_model = GaussianMeanModel(1.0, squared=True)
    gpu_model = GaussianMeanModel(1.0, squared=True).cuda()
    cpu_layout = ParameterLayout.from_module(cpu_model)
    gpu_layout = ParameterLayout.from_module(gpu_model)
    cpu_inputs, cpu_targets, cpu_direction = _toy_batch(cpu_model)
    gpu_inputs, gpu_targets, gpu_direction = _toy_batch(gpu_model)

    cpu = per_sample_derivatives(
        cpu_model,
        cpu_inputs,
        cpu_targets,
        gaussian_nll,
        cpu_layout,
        direction=cpu_direction,
        strategy="vmap",
    )
    gpu = per_sample_derivatives(
        gpu_model,
        gpu_inputs,
        gpu_targets,
        gaussian_nll,
        gpu_layout,
        direction=gpu_direction,
        strategy="vmap",
    )

    torch.testing.assert_close(gpu.gradients.cpu(), cpu.gradients)
    torch.testing.assert_close(gpu.hvps.cpu(), cpu.hvps)
