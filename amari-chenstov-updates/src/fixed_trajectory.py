"""Immutable fixed trajectories and dense auxiliary-Fisher replay."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .config import OptimizerConfig
from .ewc import build_optimizer, take_ewc_proposal
from .fisher import LFUBatchEstimate
from .mnist_data import MixtureStreamPlan
from .parameters import ParameterLayout
from .representations import PSDProjectionDiagnostics, project_psd_frobenius

FIXED_TRAJECTORY_SCHEMA_VERSION = 1
DENSE_TRACKING_SCHEMA_VERSION = 1
DENSE_METHODS = ("ema", "ac_only", "full_lfu", "periodic_fresh")


def _tensor_hash(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _materialize_batch(
    dataset: Dataset,
    observation_indices: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    observations = [dataset[int(index)] for index in observation_indices]
    inputs = torch.stack([item[0] for item in observations]).to(
        device=device,
        dtype=dtype,
    )
    targets = torch.as_tensor(
        [int(item[1]) for item in observations],
        dtype=torch.long,
        device=device,
    )
    return inputs, targets


@dataclasses.dataclass(frozen=True)
class FixedTrajectory:
    parameters: Tensor
    displacements: Tensor
    p_values: tuple[float, ...]
    observation_indices: tuple[tuple[int, ...], ...]
    class_labels: tuple[tuple[int, ...], ...]
    driver_steps: tuple[dict[str, Any], ...]
    optimizer_state: dict[str, Any]
    parameter_layout: dict[str, Any]
    stream_plan_hash: str
    driver_fisher_cadence: int
    schema_version: int = FIXED_TRAJECTORY_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != FIXED_TRAJECTORY_SCHEMA_VERSION:
            raise ValueError("unsupported fixed-trajectory schema")
        step_count = len(self.p_values)
        if self.parameters.ndim != 2 or self.parameters.shape[0] != step_count:
            raise ValueError("trajectory parameters must have shape (steps, parameters)")
        expected_displacement_shape = (
            max(step_count - 1, 0),
            self.parameters.shape[1],
        )
        if tuple(self.displacements.shape) != expected_displacement_shape:
            raise ValueError("trajectory displacements have an invalid shape")
        if step_count > 1 and not torch.equal(
            self.displacements,
            self.parameters[1:] - self.parameters[:-1],
        ):
            raise ValueError("trajectory displacements do not match its parameters")
        if (
            len(self.observation_indices) != step_count
            or len(self.class_labels) != step_count
            or len(self.driver_steps) != step_count
        ):
            raise ValueError("trajectory metadata does not align by step")
        if self.driver_fisher_cadence < 1:
            raise ValueError("driver Fisher cadence must be positive")

    @property
    def content_hash(self) -> str:
        self.validate()
        metadata = {
            "schema_version": self.schema_version,
            "parameters_hash": _tensor_hash(self.parameters),
            "displacements_hash": _tensor_hash(self.displacements),
            "p_values": self.p_values,
            "observation_indices": self.observation_indices,
            "class_labels": self.class_labels,
            "driver_steps": self.driver_steps,
            "parameter_layout": self.parameter_layout,
            "stream_plan_hash": self.stream_plan_hash,
            "driver_fisher_cadence": self.driver_fisher_cadence,
        }
        return _canonical_hash(metadata)

    def artifact_mapping(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "parameters": self.parameters,
            "displacements": self.displacements,
            "p_values": self.p_values,
            "observation_indices": self.observation_indices,
            "class_labels": self.class_labels,
            "driver_steps": self.driver_steps,
            "optimizer_state": self.optimizer_state,
            "parameter_layout": self.parameter_layout,
            "stream_plan_hash": self.stream_plan_hash,
            "driver_fisher_cadence": self.driver_fisher_cadence,
        }


DriverFisherProvider = Callable[
    [int, float, nn.Module, ParameterLayout],
    tuple[Tensor, dict[str, Any]],
]


def generate_fixed_trajectory(
    model: nn.Module,
    layout: ParameterLayout,
    dataset: Dataset,
    stream_plan: MixtureStreamPlan,
    optimizer_config: OptimizerConfig,
    *,
    fisher_cadence: int,
    fisher_provider: DriverFisherProvider,
    device: torch.device,
    dtype: torch.dtype,
) -> FixedTrajectory:
    """Generate one oracle-assisted path before any Fisher treatment is replayed."""

    if fisher_cadence < 1:
        raise ValueError("fisher_cadence must be positive")
    stream_plan.validate()
    layout.validate_module(model)
    optimizer = build_optimizer(model, optimizer_config)
    parameters = []
    displacements = []
    driver_steps = []
    driver_fisher = None

    for step, p_value in enumerate(stream_plan.p_values):
        parameter_before = layout.flatten_module(model, detach=True)
        parameters.append(parameter_before.cpu())
        refreshed = step % fisher_cadence == 0
        refresh_metadata = None
        if refreshed:
            driver_fisher, refresh_metadata = fisher_provider(
                step,
                p_value,
                model,
                layout,
            )
            if driver_fisher.shape != (layout.total_numel, layout.total_numel):
                raise ValueError("driver Fisher has an invalid shape")
            driver_fisher = driver_fisher.to(device=device, dtype=dtype)
        if driver_fisher is None:
            raise RuntimeError("the driver requires a Fisher at its first step")

        inputs, targets = _materialize_batch(
            dataset,
            stream_plan.observation_indices[step],
            device=device,
            dtype=dtype,
        )
        expected_labels = torch.as_tensor(
            stream_plan.class_labels[step],
            dtype=torch.long,
            device=device,
        )
        if not torch.equal(targets, expected_labels):
            raise ValueError("stream-plan labels do not match the dataset")

        if step + 1 == len(stream_plan.p_values):
            driver_steps.append(
                {
                    "step": step,
                    "p": p_value,
                    "fisher_refreshed": refreshed,
                    "fisher_refresh": refresh_metadata,
                    "proposal": None,
                }
            )
            continue

        proposal = take_ewc_proposal(
            model,
            layout,
            inputs,
            targets,
            driver_fisher,
            optimizer_config,
            optimizer,
        )
        displacement = layout.flatten_module(model, detach=True) - parameter_before
        if not torch.equal(displacement.cpu(), proposal.displacement):
            raise RuntimeError("recorded proposal does not equal the accepted move")
        displacements.append(displacement.cpu())
        driver_steps.append(
            {
                "step": step,
                "p": p_value,
                "fisher_refreshed": refreshed,
                "fisher_refresh": refresh_metadata,
                "proposal": proposal.metrics_mapping(),
            }
        )

    trajectory = FixedTrajectory(
        parameters=torch.stack(parameters),
        displacements=torch.stack(displacements),
        p_values=stream_plan.p_values,
        observation_indices=stream_plan.observation_indices,
        class_labels=stream_plan.class_labels,
        driver_steps=tuple(driver_steps),
        optimizer_state=optimizer.state_dict(),
        parameter_layout=layout.metadata(),
        stream_plan_hash=stream_plan.content_hash,
        driver_fisher_cadence=fisher_cadence,
    )
    trajectory.validate()
    return trajectory


@dataclasses.dataclass(frozen=True)
class OnlineStepStatistics:
    estimate: LFUBatchEstimate
    score_gradient_count: int
    hvp_count: int
    elapsed_seconds: float

    def validate(self, parameter_count: int) -> None:
        expected = (parameter_count, parameter_count)
        for name in ("fisher", "amari_chentsov", "residual"):
            matrix = getattr(self.estimate, name)
            if matrix.shape != expected:
                raise ValueError(f"{name} has an invalid shape")
            if not torch.isfinite(matrix).all():
                raise ValueError(f"{name} contains nonfinite values")
        if self.score_gradient_count < 1 or self.hvp_count < 0:
            raise ValueError("derivative counts must be nonnegative")


def lagged_directions(
    trajectory: FixedTrajectory,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[Tensor]:
    """Return zero at ``t=0`` followed by the realized ``u_{t-1}`` rows."""

    trajectory.validate()
    return [
        (
            torch.zeros(
                trajectory.parameters.shape[1],
                device=device,
                dtype=dtype,
            )
            if step == 0
            else trajectory.displacements[step - 1].to(
                device=device,
                dtype=dtype,
            )
        )
        for step in range(len(trajectory.p_values))
    ]


@dataclasses.dataclass(frozen=True)
class DenseConditionResult:
    method: str
    estimates: tuple[Tensor, ...]
    predictions: tuple[Tensor, ...]
    candidates: tuple[Tensor, ...]
    metrics: tuple[dict[str, Any], ...]


def _relative_frobenius(estimate: Tensor, target: Tensor) -> float:
    denominator = torch.linalg.matrix_norm(target, ord="fro")
    floor = torch.finfo(target.dtype).eps
    return float(
        torch.linalg.matrix_norm(estimate - target, ord="fro")
        / denominator.clamp_min(floor)
    )


def _operator_norm(matrix: Tensor) -> float:
    symmetric = (matrix + matrix.mT) / 2
    return float(torch.linalg.eigvalsh(symmetric).abs().max())


def _projection_mapping(diagnostics: PSDProjectionDiagnostics) -> dict[str, Any]:
    return dataclasses.asdict(diagnostics)


def _matrix_metrics(
    estimate: Tensor,
    prediction: Tensor,
    candidate: Tensor,
    reference: Tensor,
    direction: Tensor,
    projection: PSDProjectionDiagnostics,
    *,
    method: str,
    step: int,
    p_value: float,
    refreshed: bool,
    correction: Tensor,
) -> dict[str, Any]:
    error = estimate - reference
    reference_eigenvalues, reference_eigenvectors = torch.linalg.eigh(reference)
    estimate_eigenvalues, estimate_eigenvectors = torch.linalg.eigh(estimate)
    leading_alignment = torch.abs(
        reference_eigenvectors[:, -1] @ estimate_eigenvectors[:, -1]
    )
    directional_error = direction @ (error @ direction)
    return {
        "method": method,
        "step": step,
        "p": p_value,
        "fresh_replacement": refreshed,
        "estimate_fro": float(torch.linalg.matrix_norm(estimate, ord="fro")),
        "absolute_frobenius_error": float(
            torch.linalg.matrix_norm(error, ord="fro")
        ),
        "relative_frobenius_error": _relative_frobenius(estimate, reference),
        "prediction_relative_frobenius_error": _relative_frobenius(
            prediction,
            reference,
        ),
        "candidate_relative_frobenius_error": _relative_frobenius(
            candidate,
            reference,
        ),
        "operator_norm_error": _operator_norm(error),
        "directional_error": float(directional_error),
        "absolute_directional_error": float(torch.abs(directional_error)),
        "leading_eigenvalue": float(estimate_eigenvalues[-1]),
        "reference_leading_eigenvalue": float(reference_eigenvalues[-1]),
        "leading_eigenvector_alignment": float(leading_alignment),
        "applied_correction_fro": float(
            torch.linalg.matrix_norm(correction, ord="fro")
        ),
        "projection": _projection_mapping(projection),
    }


def common_step_metrics(
    references: Sequence[Tensor],
    statistics: Sequence[OnlineStepStatistics],
    directions: Sequence[Tensor],
    p_values: Sequence[float],
) -> list[dict[str, Any]]:
    if not (
        len(references) == len(statistics) == len(directions) == len(p_values)
    ):
        raise ValueError("common step inputs must have equal lengths")
    rows = []
    for step, (reference, statistic, direction, p_value) in enumerate(
        zip(references, statistics, directions, p_values, strict=True)
    ):
        estimate = statistic.estimate
        ac_norm = torch.linalg.matrix_norm(estimate.amari_chentsov, ord="fro")
        residual_norm = torch.linalg.matrix_norm(estimate.residual, ord="fro")
        alignment_denominator = ac_norm * residual_norm
        reference_increment = (
            None if step == 0 else reference - references[step - 1]
        )
        rows.append(
            {
                "step": step,
                "p": p_value,
                "direction_norm": float(torch.linalg.vector_norm(direction)),
                "direct_fisher_fro": float(
                    torch.linalg.matrix_norm(estimate.fisher, ord="fro")
                ),
                "direct_fisher_relative_error": _relative_frobenius(
                    estimate.fisher,
                    reference,
                ),
                "ac_fro": float(ac_norm),
                "residual_fro": float(residual_norm),
                "full_lfu_fro": float(
                    torch.linalg.matrix_norm(estimate.full, ord="fro")
                ),
                "ac_residual_alignment": (
                    None
                    if float(alignment_denominator) == 0.0
                    else float(
                        (estimate.amari_chentsov * estimate.residual).sum()
                        / alignment_denominator
                    )
                ),
                "reference_increment_fro": (
                    None
                    if reference_increment is None
                    else float(
                        torch.linalg.matrix_norm(
                            reference_increment,
                            ord="fro",
                        )
                    )
                ),
                "ac_increment_relative_error": (
                    None
                    if reference_increment is None
                    else _relative_frobenius(
                        estimate.amari_chentsov,
                        reference_increment,
                    )
                ),
                "full_increment_relative_error": (
                    None
                    if reference_increment is None
                    else _relative_frobenius(
                        estimate.full,
                        reference_increment,
                    )
                ),
                "score_gradient_count": statistic.score_gradient_count,
                "hvp_count": statistic.hvp_count,
                "derivative_elapsed_seconds": statistic.elapsed_seconds,
            }
        )
    return rows


def replay_dense_conditions(
    initial_fisher: Tensor,
    references: Sequence[Tensor],
    statistics: Sequence[OnlineStepStatistics],
    directions: Sequence[Tensor],
    p_values: Sequence[float],
    *,
    ema_gain: float,
    fresh_fisher_cadence: int,
) -> dict[str, DenseConditionResult]:
    """Replay all dense estimators without allowing them to mutate the path."""

    if not 0.0 < ema_gain <= 1.0:
        raise ValueError("ema_gain must be in (0, 1]")
    if fresh_fisher_cadence < 1:
        raise ValueError("fresh_fisher_cadence must be positive")
    step_count = len(references)
    if not (
        step_count == len(statistics) == len(directions) == len(p_values)
        and step_count > 0
    ):
        raise ValueError("replay inputs must have one aligned row per step")
    parameter_count = initial_fisher.shape[0]
    if initial_fisher.shape != (parameter_count, parameter_count):
        raise ValueError("initial Fisher must be square")
    for statistic in statistics:
        statistic.validate(parameter_count)

    results = {}
    for method in DENSE_METHODS:
        previous = project_psd_frobenius(initial_fisher).projected
        estimates = []
        predictions = []
        candidates = []
        metrics = []
        for step in range(step_count):
            if step == 0:
                correction = torch.zeros_like(previous)
                prediction = previous
                candidate = previous
                projection = project_psd_frobenius(candidate)
                current = projection.projected
                refreshed = False
            else:
                if method in {"ema", "periodic_fresh"}:
                    correction = torch.zeros_like(previous)
                elif method == "ac_only":
                    correction = statistics[step].estimate.amari_chentsov
                else:
                    correction = statistics[step].estimate.full
                prediction = previous + correction
                candidate = (
                    (1.0 - ema_gain) * prediction
                    + ema_gain * statistics[step].estimate.fisher
                )
                refreshed = (
                    method == "periodic_fresh"
                    and step % fresh_fisher_cadence == 0
                )
                projection_source = references[step] if refreshed else candidate
                projection = project_psd_frobenius(projection_source)
                current = projection.projected

            metrics.append(
                _matrix_metrics(
                    current,
                    prediction,
                    candidate,
                    references[step],
                    directions[step],
                    projection.diagnostics,
                    method=method,
                    step=step,
                    p_value=float(p_values[step]),
                    refreshed=refreshed,
                    correction=correction,
                )
            )
            estimates.append(current.detach().cpu())
            predictions.append(prediction.detach().cpu())
            candidates.append(candidate.detach().cpu())
            previous = current

        results[method] = DenseConditionResult(
            method=method,
            estimates=tuple(estimates),
            predictions=tuple(predictions),
            candidates=tuple(candidates),
            metrics=tuple(metrics),
        )
    return results


def timed_online_statistics(
    estimate_function: Callable[[], LFUBatchEstimate],
    *,
    score_gradient_count: int,
    hvp_count: int,
    synchronize: Callable[[], None] | None = None,
) -> OnlineStepStatistics:
    """Time one already-defined online derivative calculation."""

    sync = synchronize or (lambda: None)
    sync()
    started = time.perf_counter()
    estimate = estimate_function()
    sync()
    return OnlineStepStatistics(
        estimate=estimate,
        score_gradient_count=score_gradient_count,
        hvp_count=hvp_count,
        elapsed_seconds=time.perf_counter() - started,
    )
