"""Per-sample loss gradients and matrix-free Hessian-vector products."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.func import functional_call, grad, vmap

from .parameters import ParameterLayout

LossFunction = Callable[[Tensor, Tensor], Tensor]


@dataclasses.dataclass(frozen=True)
class PerSampleDerivatives:
    gradients: Tensor
    hvps: Tensor | None

    def validate(self, parameter_count: int) -> None:
        if self.gradients.ndim != 2 or self.gradients.shape[1] != parameter_count:
            raise ValueError(
                "gradients must have shape (batch_size, parameter_count)"
            )
        if self.hvps is not None and self.hvps.shape != self.gradients.shape:
            raise ValueError("hvps must have the same shape as gradients")


def _scalar_loss(
    loss_function: LossFunction,
    predictions: Tensor,
    targets: Tensor,
) -> Tensor:
    loss = loss_function(predictions, targets)
    if loss.numel() != 1:
        raise ValueError("loss_function must return one scalar per sample call")
    return loss.reshape(())


def _validate_inputs(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    layout: ParameterLayout,
    direction: Tensor | None,
) -> None:
    layout.validate_module(model)
    if inputs.ndim < 1 or targets.ndim < 1:
        raise ValueError("inputs and targets must have a batch dimension")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have equal batch sizes")
    if inputs.shape[0] == 0:
        raise ValueError("cannot calculate derivatives for an empty batch")
    if direction is not None and (
        direction.ndim != 1 or direction.numel() != layout.total_numel
    ):
        raise ValueError(
            f"direction must have shape ({layout.total_numel},), "
            f"got {tuple(direction.shape)}"
        )


def _loop_derivatives(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    loss_function: LossFunction,
    layout: ParameterLayout,
    direction: Tensor | None,
) -> PerSampleDerivatives:
    named_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    parameters = tuple(named_parameters[name] for name in layout.names)
    direction_named = (
        None if direction is None else layout.unflatten_named(direction)
    )

    gradient_rows = []
    hvp_rows = []
    for index in range(inputs.shape[0]):
        predictions = model(inputs[index : index + 1])
        loss = _scalar_loss(
            loss_function,
            predictions,
            targets[index : index + 1],
        )
        gradients = torch.autograd.grad(
            loss,
            parameters,
            create_graph=direction is not None,
        )
        gradient_named = dict(zip(layout.names, gradients, strict=True))
        gradient_rows.append(layout.flatten_named(gradient_named))

        if direction_named is not None:
            gradient_dot_direction = sum(
                (
                    gradient_named[name] * direction_named[name]
                ).sum()
                for name in layout.names
            )
            hvps = torch.autograd.grad(gradient_dot_direction, parameters)
            hvp_rows.append(
                layout.flatten_named(
                    dict(zip(layout.names, hvps, strict=True))
                )
            )

    derivatives = PerSampleDerivatives(
        gradients=torch.stack(gradient_rows).detach(),
        hvps=None if direction is None else torch.stack(hvp_rows).detach(),
    )
    derivatives.validate(layout.total_numel)
    return derivatives


def _vmap_derivatives(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    loss_function: LossFunction,
    layout: ParameterLayout,
    direction: Tensor | None,
) -> PerSampleDerivatives:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    buffers = dict(model.named_buffers())

    def one_loss(
        parameter_values: dict[str, Tensor],
        one_input: Tensor,
        one_target: Tensor,
    ) -> Tensor:
        predictions = functional_call(
            model,
            (parameter_values, buffers),
            (one_input.unsqueeze(0),),
        )
        return _scalar_loss(
            loss_function,
            predictions,
            one_target.unsqueeze(0),
        )

    one_gradient = grad(one_loss)
    if direction is None:
        gradient_named = vmap(one_gradient, in_dims=(None, 0, 0))(
            parameters,
            inputs,
            targets,
        )
        derivatives = PerSampleDerivatives(
            gradients=layout.flatten_batched_named(gradient_named).detach(),
            hvps=None,
        )
        derivatives.validate(layout.total_numel)
        return derivatives

    direction_named = layout.unflatten_named(direction)

    def one_gradient_and_hvp(
        parameter_values: dict[str, Tensor],
        one_input: Tensor,
        one_target: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        gradient_values = one_gradient(
            parameter_values,
            one_input,
            one_target,
        )

        def gradient_dot_direction(
            differentiable_parameters: dict[str, Tensor],
        ) -> Tensor:
            current_gradient = one_gradient(
                differentiable_parameters,
                one_input,
                one_target,
            )
            return sum(
                (
                    current_gradient[name] * direction_named[name]
                ).sum()
                for name in layout.names
            )

        hvp_values = grad(gradient_dot_direction)(parameter_values)
        return gradient_values, hvp_values

    gradient_named, hvp_named = vmap(
        one_gradient_and_hvp,
        in_dims=(None, 0, 0),
    )(parameters, inputs, targets)
    derivatives = PerSampleDerivatives(
        gradients=layout.flatten_batched_named(gradient_named).detach(),
        hvps=layout.flatten_batched_named(hvp_named).detach(),
    )
    derivatives.validate(layout.total_numel)
    return derivatives


def per_sample_derivatives(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    loss_function: LossFunction,
    layout: ParameterLayout,
    *,
    direction: Tensor | None = None,
    strategy: str = "loop",
) -> PerSampleDerivatives:
    """Calculate per-sample gradients and optional HVPs without `.grad` mutation."""

    _validate_inputs(model, inputs, targets, layout, direction)
    if strategy == "loop":
        return _loop_derivatives(
            model,
            inputs,
            targets,
            loss_function,
            layout,
            direction,
        )
    if strategy == "vmap":
        return _vmap_derivatives(
            model,
            inputs,
            targets,
            loss_function,
            layout,
            direction,
        )
    raise ValueError("strategy must be 'loop' or 'vmap'")
