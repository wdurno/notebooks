"""Affine-coordinate strong-Wolfe L-BFGS for repaired Plan 13 trajectories."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call

from src.parameters import ParameterLayout

from .geometry import AdaptationGeometry


@dataclasses.dataclass(frozen=True)
class CoordinateFactor:
    """A matrix-free factor ``A`` mapping local coordinates to parameters."""

    kind: str
    parameter_count: int
    coordinate_count: int
    basis: Tensor | None = None
    parallel_scales: Tensor | None = None
    orthogonal_scale: float = 0.0
    selected_indices: Tensor | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"identity", "selection", "low_rank", "spectral_full"}:
            raise ValueError(f"unsupported coordinate factor: {self.kind}")
        if self.parameter_count < 1 or self.coordinate_count < 1:
            raise ValueError("coordinate dimensions must be positive")
        if not math.isfinite(self.orthogonal_scale) or self.orthogonal_scale < 0:
            raise ValueError("orthogonal scale must be finite and nonnegative")
        if self.kind == "identity":
            if self.coordinate_count != self.parameter_count:
                raise ValueError("identity factor must be square")
            return
        if self.kind == "selection":
            if self.selected_indices is None:
                raise ValueError("selection factor requires selected indices")
            if self.selected_indices.shape != (self.coordinate_count,):
                raise ValueError("selection indices have the wrong shape")
            if self.selected_indices.dtype != torch.long:
                raise ValueError("selection indices must be int64")
            if (
                int(self.selected_indices.min()) < 0
                or int(self.selected_indices.max()) >= self.parameter_count
                or torch.unique(self.selected_indices).numel() != self.coordinate_count
            ):
                raise ValueError("selection indices must be unique and in range")
            return
        if self.basis is None or self.parallel_scales is None:
            raise ValueError("spectral factors require a basis and scales")
        rank = self.basis.shape[1]
        if self.basis.shape[0] != self.parameter_count or self.parallel_scales.shape != (rank,):
            raise ValueError("spectral factor dimensions are inconsistent")
        if not torch.isfinite(self.basis).all() or not torch.isfinite(self.parallel_scales).all():
            raise ValueError("spectral factor must be finite")
        if torch.any(self.parallel_scales < 0):
            raise ValueError("spectral scales must be nonnegative")
        identity = torch.eye(rank, device=self.basis.device, dtype=self.basis.dtype)
        tolerance = 256 * torch.finfo(self.basis.dtype).eps * self.parameter_count
        if not torch.allclose(
            self.basis.mT @ self.basis,
            identity,
            atol=tolerance,
            rtol=tolerance,
        ):
            raise ValueError("spectral factor basis must be orthonormal")
        expected = rank if self.kind == "low_rank" else self.parameter_count
        if self.coordinate_count != expected:
            raise ValueError("spectral coordinate count is inconsistent")

    @property
    def device(self) -> torch.device:
        if self.basis is not None:
            return self.basis.device
        if self.selected_indices is not None:
            return self.selected_indices.device
        return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype | None:
        if self.basis is not None:
            return self.basis.dtype
        return None

    def apply(self, coordinates: Tensor) -> Tensor:
        if coordinates.shape != (self.coordinate_count,):
            raise ValueError("coordinate vector has the wrong shape")
        if self.kind == "identity":
            return coordinates
        if self.kind == "selection":
            assert self.selected_indices is not None
            result = coordinates.new_zeros(self.parameter_count)
            return result.index_copy(0, self.selected_indices, coordinates)
        assert self.basis is not None and self.parallel_scales is not None
        if self.kind == "low_rank":
            return self.basis @ (self.parallel_scales * coordinates)
        projection = self.basis.mT @ coordinates
        correction = (self.parallel_scales - float(self.orthogonal_scale)) * projection
        return float(self.orthogonal_scale) * coordinates + self.basis @ correction

    def dense(self) -> Tensor:
        if self.kind == "identity":
            return torch.eye(self.parameter_count)
        if self.kind == "selection":
            assert self.selected_indices is not None
            result = torch.zeros(
                self.parameter_count,
                self.coordinate_count,
                device=self.selected_indices.device,
            )
            result[self.selected_indices, torch.arange(self.coordinate_count, device=result.device)] = 1
            return result
        assert self.basis is not None
        if self.kind == "low_rank":
            assert self.parallel_scales is not None
            return self.basis * self.parallel_scales
        identity = torch.eye(
            self.parameter_count,
            device=self.basis.device,
            dtype=self.basis.dtype,
        )
        assert self.parallel_scales is not None
        return float(self.orthogonal_scale) * identity + (
            self.basis
            * (self.parallel_scales - float(self.orthogonal_scale))
        ) @ self.basis.mT

    def mapping(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parameter_count": self.parameter_count,
            "coordinate_count": self.coordinate_count,
            "effective_rank": self.coordinate_count,
            "orthogonal_scale": self.orthogonal_scale,
            "parallel_scales": (
                None
                if self.parallel_scales is None
                else [float(value) for value in self.parallel_scales]
            ),
            "selected_indices": (
                None
                if self.selected_indices is None
                else [int(value) for value in self.selected_indices]
            ),
        }


def identity_factor(
    parameter_count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> CoordinateFactor:
    # Device and dtype are represented by the optimized coordinate itself.
    del device, dtype
    return CoordinateFactor("identity", parameter_count, parameter_count)


def selection_factor(mask: Tensor) -> CoordinateFactor:
    if mask.ndim != 1 or not torch.isfinite(mask).all():
        raise ValueError("selection mask must be a finite vector")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("selection mask must be binary")
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        raise ValueError("selection factor cannot be empty")
    return CoordinateFactor(
        "selection",
        mask.numel(),
        indices.numel(),
        selected_indices=indices,
    )


def geometry_factor(
    geometry: AdaptationGeometry,
    *,
    alpha: float,
    epsilon: float,
    projector_only: bool,
) -> CoordinateFactor:
    alpha_value = float(alpha)
    epsilon_value = float(epsilon)
    if not 0 <= alpha_value <= 1 or epsilon_value < 0:
        raise ValueError("alpha must lie in [0, 1] and epsilon must be nonnegative")
    learned = (
        torch.ones_like(geometry.eigenvalues)
        if projector_only
        else geometry.normalized_eigenvalues()
    )
    parallel_eigenvalues = (1 - alpha_value) + alpha_value * learned
    orthogonal_eigenvalue = (1 - alpha_value) + alpha_value * epsilon_value
    tolerance = 64 * torch.finfo(geometry.basis.dtype).eps
    if orthogonal_eigenvalue <= tolerance:
        keep = parallel_eigenvalues > tolerance
        if not bool(keep.any()):
            raise ValueError("geometry factor has no nonzero directions")
        basis = geometry.basis[:, keep]
        scales = parallel_eigenvalues[keep].sqrt()
        return CoordinateFactor(
            "low_rank",
            geometry.parameter_count,
            int(keep.sum()),
            basis=basis,
            parallel_scales=scales,
        )
    return CoordinateFactor(
        "spectral_full",
        geometry.parameter_count,
        geometry.parameter_count,
        basis=geometry.basis,
        parallel_scales=parallel_eigenvalues.clamp_min(0).sqrt(),
        orthogonal_scale=math.sqrt(orthogonal_eigenvalue),
    )


@dataclasses.dataclass(frozen=True)
class CoordinateLBFGSResult:
    displacement: Tensor
    objective_before: float
    objective_after: float
    data_loss_before: float
    data_loss_after: float
    penalty_after: float
    coordinate_gradient_norm_before: float
    coordinate_gradient_norm_after: float
    parameter_gradient_norm_before: float
    parameter_gradient_norm_after: float
    optimizer_iterations: int
    optimizer_function_evaluations: int
    displacement_norm: float
    fisher_weighted_displacement_norm: float
    objective_decrease: float
    stopping_reason: str
    factor: dict[str, Any]

    def mapping(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        value.pop("displacement")
        value["relative_final_coordinate_gradient_norm"] = (
            self.coordinate_gradient_norm_after
            / max(self.coordinate_gradient_norm_before, torch.finfo(torch.float64).tiny)
        )
        value["relative_final_parameter_gradient_norm"] = (
            self.parameter_gradient_norm_after
            / max(self.parameter_gradient_norm_before, torch.finfo(torch.float64).tiny)
        )
        return value


def _functional_objective(
    model: nn.Module,
    layout: ParameterLayout,
    parameter_vector: Tensor,
    buffers: Mapping[str, Tensor],
    inputs: Tensor,
    targets: Tensor,
    anchor: Tensor,
    fisher: Any,
    strength: float,
) -> tuple[Tensor, Tensor, Tensor]:
    parameters = layout.unflatten_named(parameter_vector)
    logits = functional_call(model, (parameters, buffers), (inputs,), strict=True)
    data_loss = nn.functional.cross_entropy(logits, targets)
    displacement = parameter_vector - anchor
    quadratic = displacement @ fisher.matvec(displacement)
    penalty = 0.5 * float(strength) * quadratic
    objective = data_loss + penalty
    if not torch.isfinite(objective):
        raise RuntimeError("coordinate L-BFGS objective became nonfinite")
    return objective, data_loss, penalty


def coordinate_objective_and_gradient(
    model: nn.Module,
    layout: ParameterLayout,
    coordinates: Tensor,
    factor: CoordinateFactor,
    inputs: Tensor,
    targets: Tensor,
    fisher: Any,
    *,
    strength: float,
    anchor: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return the objective and coordinate gradient without mutating ``model``."""

    base = layout.flatten_module(model, detach=True) if anchor is None else anchor
    variable = coordinates.detach().clone().requires_grad_(True)
    vector = base + factor.apply(variable)
    buffers = dict(model.named_buffers())
    objective, _, _ = _functional_objective(
        model,
        layout,
        vector,
        buffers,
        inputs,
        targets,
        base,
        fisher,
        strength,
    )
    gradient = torch.autograd.grad(objective, variable)[0]
    return objective.detach(), gradient.detach()


def coordinate_lbfgs_update(
    model: nn.Module,
    layout: ParameterLayout,
    inputs: Tensor,
    targets: Tensor,
    fisher: Any,
    factor: CoordinateFactor,
    *,
    strength: float,
    learning_rate: float = 1.0,
    inner_steps: int = 50,
    max_eval: int = 75,
    history_size: int = 20,
    tolerance_grad: float = 1e-5,
    tolerance_change: float = 1e-9,
) -> CoordinateLBFGSResult:
    """Minimize one retained objective in the declared affine coordinates."""

    layout.validate_module(model)
    anchor = layout.flatten_module(model, detach=True)
    if factor.parameter_count != layout.total_numel:
        raise ValueError("coordinate factor and model dimension differ")
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("EWC strength must be finite and nonnegative")
    if max_eval < inner_steps:
        raise ValueError("maximum evaluations must be at least the iteration budget")
    coordinates = nn.Parameter(
        torch.zeros(
            factor.coordinate_count,
            device=anchor.device,
            dtype=anchor.dtype,
        )
    )
    optimizer = torch.optim.LBFGS(
        [coordinates],
        lr=learning_rate,
        max_iter=inner_steps,
        max_eval=max_eval,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=history_size,
        line_search_fn="strong_wolfe",
    )
    buffers = dict(model.named_buffers())

    def evaluate(*, backward: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if backward:
            optimizer.zero_grad(set_to_none=True)
        vector = anchor + factor.apply(coordinates)
        objective, data_loss, penalty = _functional_objective(
            model,
            layout,
            vector,
            buffers,
            inputs,
            targets,
            anchor,
            fisher,
            strength,
        )
        if backward:
            objective.backward()
        return objective, data_loss, penalty, vector

    objective_before_tensor, data_before_tensor, _, _ = evaluate(backward=True)
    coordinate_gradient_before = coordinates.grad.detach().clone()

    parameter_probe = anchor.detach().clone().requires_grad_(True)
    parameter_objective, _, _ = _functional_objective(
        model,
        layout,
        parameter_probe,
        buffers,
        inputs,
        targets,
        anchor,
        fisher,
        strength,
    )
    parameter_gradient_before = torch.autograd.grad(parameter_objective, parameter_probe)[0]
    original = anchor.clone()

    def closure() -> Tensor:
        objective, _, _, _ = evaluate(backward=True)
        if coordinates.grad is None or not torch.isfinite(coordinates.grad).all():
            raise RuntimeError("coordinate L-BFGS produced an invalid gradient")
        return objective

    try:
        optimizer.step(closure)
        final_objective, final_data, final_penalty, final_vector = evaluate(backward=True)
        coordinate_gradient_after = coordinates.grad.detach().clone()
        final_probe = final_vector.detach().clone().requires_grad_(True)
        final_parameter_objective, _, _ = _functional_objective(
            model,
            layout,
            final_probe,
            buffers,
            inputs,
            targets,
            anchor,
            fisher,
            strength,
        )
        parameter_gradient_after = torch.autograd.grad(
            final_parameter_objective,
            final_probe,
        )[0]
        tolerance = (
            64
            * torch.finfo(anchor.dtype).eps
            * max(abs(float(objective_before_tensor.detach())), 1.0)
        )
        if float(final_objective.detach()) > float(objective_before_tensor.detach()) + tolerance:
            raise RuntimeError("coordinate L-BFGS increased the retained objective")
        if not torch.isfinite(final_vector).all():
            raise RuntimeError("coordinate L-BFGS produced nonfinite parameters")
        layout.copy_vector_to_module(model, final_vector.detach())
    except BaseException:
        layout.copy_vector_to_module(model, original)
        raise

    state = optimizer.state.get(coordinates, {})
    iterations = int(state.get("n_iter", 0))
    evaluations = int(state.get("func_evals", 0))
    if iterations >= inner_steps:
        stopping_reason = "lbfgs_max_iterations"
    elif evaluations >= max_eval:
        stopping_reason = "lbfgs_max_evaluations"
    else:
        stopping_reason = "lbfgs_convergence_tolerance"
    displacement = final_vector.detach() - anchor
    quadratic = displacement @ fisher.matvec(displacement)
    numerical_floor = -100 * torch.finfo(quadratic.dtype).eps
    if float(quadratic) < numerical_floor:
        layout.copy_vector_to_module(model, original)
        raise RuntimeError("PSD Fisher produced a materially negative quadratic")
    values = (
        float(objective_before_tensor.detach()),
        float(final_objective.detach()),
        float(data_before_tensor.detach()),
        float(final_data.detach()),
        float(final_penalty.detach()),
        float(torch.linalg.vector_norm(coordinate_gradient_before)),
        float(torch.linalg.vector_norm(coordinate_gradient_after)),
        float(torch.linalg.vector_norm(parameter_gradient_before)),
        float(torch.linalg.vector_norm(parameter_gradient_after)),
        float(torch.linalg.vector_norm(displacement)),
        float(torch.sqrt(quadratic.clamp_min(0))),
    )
    if not all(math.isfinite(value) for value in values):
        layout.copy_vector_to_module(model, original)
        raise RuntimeError("coordinate L-BFGS produced nonfinite diagnostics")
    return CoordinateLBFGSResult(
        displacement=displacement.detach().cpu(),
        objective_before=values[0],
        objective_after=values[1],
        data_loss_before=values[2],
        data_loss_after=values[3],
        penalty_after=values[4],
        coordinate_gradient_norm_before=values[5],
        coordinate_gradient_norm_after=values[6],
        parameter_gradient_norm_before=values[7],
        parameter_gradient_norm_after=values[8],
        optimizer_iterations=iterations,
        optimizer_function_evaluations=evaluations,
        displacement_norm=values[9],
        fisher_weighted_displacement_norm=values[10],
        objective_decrease=values[0] - values[1],
        stopping_reason=stopping_reason,
        factor=factor.mapping(),
    )


__all__ = [
    "CoordinateFactor",
    "CoordinateLBFGSResult",
    "coordinate_lbfgs_update",
    "coordinate_objective_and_gradient",
    "geometry_factor",
    "identity_factor",
    "selection_factor",
]
