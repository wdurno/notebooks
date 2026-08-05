"""Warm-started high-sample reference-optimum paths for controller diagnostics."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import time
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .convergence import HilbertMean
from .mnist_data import (
    DatasetPartitions,
    ReferenceSamplePlan,
    generate_reference_sample_plan,
)
from .parameters import ParameterLayout
from .seeding import derive_component_seed

REFERENCE_OPTIMUM_PATH_SCHEMA_VERSION = 2


def _tensor_digest(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class ReferenceOptimumPath:
    p_values: tuple[float, ...]
    parameters: Tensor
    displacements: Tensor
    rows: tuple[dict[str, Any], ...]
    sample_plans: tuple[ReferenceSamplePlan, ...]
    content_hash: str
    replicate_parameters: Tensor | None = None
    fit_rows: tuple[dict[str, Any], ...] = ()
    displacement_diagnostics: tuple[dict[str, Any], ...] = ()
    schema_version: int = REFERENCE_OPTIMUM_PATH_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version not in {1, REFERENCE_OPTIMUM_PATH_SCHEMA_VERSION}:
            raise ValueError("unsupported reference-optimum path schema")
        if self.parameters.ndim != 2 or not torch.isfinite(self.parameters).all():
            raise ValueError("reference parameters must be a finite matrix")
        expected_displacement_shape = (
            max(len(self.p_values) - 1, 0),
            self.parameters.shape[1],
        )
        if self.displacements.shape != expected_displacement_shape:
            raise ValueError("reference displacements have an invalid shape")
        if self.parameters.shape[0] != len(self.p_values):
            raise ValueError("reference parameter and p counts differ")
        if len(self.rows) != len(self.p_values) or len(self.sample_plans) != len(
            self.p_values
        ):
            raise ValueError("reference path metadata counts differ")
        if self.displacements.numel() and not torch.equal(
            self.displacements,
            self.parameters[1:] - self.parameters[:-1],
        ):
            raise ValueError("reference displacements do not match parameters")
        if not isinstance(self.content_hash, str) or len(self.content_hash) != 64:
            raise ValueError("reference path content hash is invalid")
        if self.replicate_parameters is not None:
            if self.replicate_parameters.ndim != 3:
                raise ValueError("reference replicate parameters must be rank three")
            if self.replicate_parameters.shape[1:] != self.parameters.shape:
                raise ValueError("reference replicate parameters have an invalid shape")
            if not torch.isfinite(self.replicate_parameters).all():
                raise ValueError("reference replicate parameters must be finite")
        if self.schema_version >= 2 and len(self.displacement_diagnostics) != len(
            self.p_values
        ):
            raise ValueError("reference displacement diagnostics are incomplete")

    def artifact_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "p_values": self.p_values,
            "parameters": self.parameters,
            "displacements": self.displacements,
            "rows": self.rows,
            "sample_plans": tuple(plan.to_mapping() for plan in self.sample_plans),
            "content_hash": self.content_hash,
            "replicate_parameters": self.replicate_parameters,
            "fit_rows": self.fit_rows,
            "displacement_diagnostics": self.displacement_diagnostics,
        }

    @classmethod
    def from_artifact_mapping(cls, value: dict[str, Any]) -> "ReferenceOptimumPath":
        """Restore and validate a path written by :meth:`artifact_mapping`."""

        path = cls(
            schema_version=int(value["schema_version"]),
            p_values=tuple(float(item) for item in value["p_values"]),
            parameters=value["parameters"],
            displacements=value["displacements"],
            rows=tuple(dict(item) for item in value["rows"]),
            sample_plans=tuple(
                ReferenceSamplePlan.from_mapping(dict(item))
                for item in value["sample_plans"]
            ),
            content_hash=str(value["content_hash"]),
            replicate_parameters=value.get("replicate_parameters"),
            fit_rows=tuple(dict(item) for item in value.get("fit_rows", ())),
            displacement_diagnostics=tuple(
                dict(item)
                for item in value.get("displacement_diagnostics", ())
            ),
        )
        path.validate()
        expected_hash = _path_hash(
            path.p_values,
            path.parameters,
            path.sample_plans,
            schema_version=path.schema_version,
            replicate_parameters=path.replicate_parameters,
        )
        if path.content_hash != expected_hash:
            raise ValueError("reference-optimum path content hash does not match")
        return path


def _path_hash(
    p_values: Sequence[float],
    parameters: Tensor,
    sample_plans: Sequence[ReferenceSamplePlan],
    *,
    schema_version: int = REFERENCE_OPTIMUM_PATH_SCHEMA_VERSION,
    replicate_parameters: Tensor | None = None,
) -> str:
    payload = {
        "schema_version": schema_version,
        "p_values": list(p_values),
        "parameter_digest": _tensor_digest(parameters),
        "sample_plan_hashes": [plan.content_hash for plan in sample_plans],
        "replicate_parameter_digest": (
            None
            if replicate_parameters is None
            else _tensor_digest(replicate_parameters)
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _build_fixed_reference_optimum_path(
    initial_model: nn.Module,
    dataset: Dataset,
    train_targets: Tensor,
    partitions: DatasetPartitions,
    p_values: Sequence[float],
    config: ExperimentConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ReferenceOptimumPath:
    """Fit a deterministic warm-started unregularized path on large batches."""

    if not p_values:
        raise ValueError("p_values cannot be empty")
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("p_values must lie in [0, 1]")
    reference = config.reference
    sample_size = reference.calibration_steps * reference.calibration_batch_size
    model = copy.deepcopy(initial_model).to(device=device, dtype=dtype)
    layout = ParameterLayout.from_module(model)
    parameters: list[Tensor] = []
    rows: list[dict[str, Any]] = []
    plans: list[ReferenceSamplePlan] = []
    previous = layout.flatten_module(model, detach=True).cpu()

    progress = tqdm(
        p_values,
        desc="reference-optimum path",
        unit="point",
        leave=False,
    )
    for step, p_value in enumerate(progress):
        seed = derive_component_seed(
            config.replica_seed,
            f"phase8_reference_optimum:step={step}:p={p_value:.17g}",
        )
        plan = generate_reference_sample_plan(
            train_targets,
            partitions,
            config.data,
            p=float(p_value),
            sample_size=sample_size,
            seed=seed,
            pool="reference",
        )
        loader = DataLoader(
            Subset(dataset, plan.observation_indices),
            batch_size=reference.calibration_batch_size,
            shuffle=False,
            num_workers=config.initialization.num_workers,
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=reference.calibration_learning_rate,
        )
        losses: list[float] = []
        gradient_norms: list[float] = []
        started = time.perf_counter()
        model.train()
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=dtype)
            targets = targets.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            squared_gradient_norm = sum(
                float(parameter.grad.detach().square().sum())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            gradient_norms.append(math.sqrt(squared_gradient_norm))
            losses.append(float(loss.detach()))
            optimizer.step()
        if len(losses) != reference.calibration_steps:
            raise RuntimeError("reference optimum used the wrong number of steps")

        current = layout.flatten_module(model, detach=True).cpu()
        parameters.append(current)
        rows.append(
            {
                "step": step,
                "p": float(p_value),
                "seed": seed,
                "sample_size": sample_size,
                "sample_plan_hash": plan.content_hash,
                "warm_started": step > 0,
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "mean_loss": sum(losses) / len(losses),
                "loss_change": losses[-1] - losses[0],
                "final_minibatch_gradient_norm": gradient_norms[-1],
                "parameter_norm": float(torch.linalg.vector_norm(current)),
                "warm_start_displacement_norm": float(
                    torch.linalg.vector_norm(current - previous)
                ),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        plans.append(plan)
        previous = current

    parameter_tensor = torch.stack(parameters)
    displacement_tensor = parameter_tensor[1:] - parameter_tensor[:-1]
    path = ReferenceOptimumPath(
        p_values=tuple(float(p) for p in p_values),
        parameters=parameter_tensor,
        displacements=displacement_tensor,
        rows=tuple(rows),
        sample_plans=tuple(plans),
        content_hash=_path_hash(
            p_values,
            parameter_tensor,
            plans,
            schema_version=1,
        ),
        schema_version=1,
    )
    path.validate()
    return path


def _fit_reference_point(
    model: nn.Module,
    dataset: Dataset,
    train_targets: Tensor,
    partitions: DatasetPartitions,
    config: ExperimentConfig,
    *,
    step: int,
    p_value: float,
    fit_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, dict[str, Any], ReferenceSamplePlan]:
    reference = config.reference
    sample_size = reference.calibration_steps * reference.calibration_batch_size
    seed = derive_component_seed(
        config.replica_seed,
        (
            "phase8_reference_optimum:"
            f"fit={fit_index}:step={step}:p={p_value:.17g}"
        ),
    )
    validation_seed = derive_component_seed(
        config.replica_seed,
        (
            "phase8_reference_optimum_validation:"
            f"fit={fit_index}:step={step}:p={p_value:.17g}"
        ),
    )
    plan = generate_reference_sample_plan(
        train_targets,
        partitions,
        config.data,
        p=float(p_value),
        sample_size=sample_size,
        seed=seed,
        pool="reference",
    )
    validation_plan = generate_reference_sample_plan(
        train_targets,
        partitions,
        config.data,
        p=float(p_value),
        sample_size=(
            reference.calibration_validation_chunks
            * reference.calibration_batch_size
        ),
        seed=validation_seed,
        pool="reference",
    )
    loader = DataLoader(
        Subset(dataset, plan.observation_indices),
        batch_size=reference.calibration_batch_size,
        shuffle=False,
        num_workers=config.initialization.num_workers,
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=reference.calibration_learning_rate,
    )
    losses: list[float] = []
    gradient_norms: list[float] = []
    started = time.perf_counter()
    model.train()
    for inputs, targets in loader:
        inputs = inputs.to(device=device, dtype=dtype)
        targets = targets.to(device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        squared_gradient_norm = sum(
            float(parameter.grad.detach().square().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        gradient_norms.append(math.sqrt(squared_gradient_norm))
        losses.append(float(loss.detach()))
        optimizer.step()
    if len(losses) != reference.calibration_steps:
        raise RuntimeError("reference optimum used the wrong number of steps")

    validation_loader = DataLoader(
        Subset(dataset, validation_plan.observation_indices),
        batch_size=reference.calibration_batch_size,
        shuffle=False,
        num_workers=config.initialization.num_workers,
    )
    gradient_moments = HilbertMean()
    validation_losses: list[float] = []
    model.eval()
    trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    for inputs, targets in validation_loader:
        inputs = inputs.to(device=device, dtype=dtype)
        targets = targets.to(device=device)
        loss = nn.functional.cross_entropy(model(inputs), targets)
        gradients = torch.autograd.grad(loss, trainable)
        gradient_moments.update(
            torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
        )
        validation_losses.append(float(loss.detach()))
    gradient_diagnostics = gradient_moments.diagnostics(
        sigma=reference.convergence_sigma,
        relative_epsilon=reference.convergence_relative_epsilon,
        absolute_epsilon=reference.convergence_absolute_epsilon,
        minimum_count=reference.calibration_validation_chunks,
        maximum_count=reference.calibration_validation_chunks,
    )
    radius = gradient_diagnostics["confidence_radius"]
    gradient_diagnostics.update(
        {
            "geometry": "euclidean_score_mean",
            "zero_in_confidence_ball": (
                radius is not None
                and gradient_diagnostics["mean_norm"] <= radius
            ),
            "stationarity_upper_bound": (
                None
                if radius is None
                else gradient_diagnostics["mean_norm"] + radius
            ),
        }
    )
    layout = ParameterLayout.from_module(model)
    current = layout.flatten_module(model, detach=True).cpu()
    training_indices = set(plan.observation_indices)
    validation_indices = set(validation_plan.observation_indices)
    return current, {
        "step": step,
        "p": float(p_value),
        "fit_index": fit_index,
        "seed": seed,
        "validation_seed": validation_seed,
        "sample_size": sample_size,
        "validation_sample_size": validation_plan.sample_size,
        "sample_plan_hash": plan.content_hash,
        "validation_plan_hash": validation_plan.content_hash,
        "warm_started": step > 0,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "loss_change": losses[-1] - losses[0],
        "final_minibatch_gradient_norm": gradient_norms[-1],
        "mean_validation_loss": sum(validation_losses) / len(validation_losses),
        "validation_score": gradient_diagnostics,
        "optimizer_steps": len(losses),
        "optimizer_stopping_reason": "maximum_budget",
        "parameter_norm": float(torch.linalg.vector_norm(current)),
        "training_draw_duplicate_fraction": (
            1.0 - len(training_indices) / plan.sample_size
        ),
        "validation_draw_duplicate_fraction": (
            1.0 - len(validation_indices) / validation_plan.sample_size
        ),
        "training_validation_unique_overlap_fraction": (
            len(training_indices & validation_indices)
            / max(len(validation_indices), 1)
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }, plan


def _mean_diagnostics(
    observations: Sequence[Tensor],
    config: ExperimentConfig,
) -> tuple[Tensor, dict[str, Any]]:
    moments = HilbertMean()
    for observation in observations:
        moments.update(observation)
    reference = config.reference
    diagnostics = moments.diagnostics(
        sigma=reference.convergence_sigma,
        relative_epsilon=reference.convergence_relative_epsilon,
        absolute_epsilon=reference.convergence_absolute_epsilon,
        minimum_count=reference.calibration_min_fits,
        maximum_count=reference.calibration_max_fits,
    )
    diagnostics["geometry"] = "euclidean"
    diagnostics["stopping_reason"] = (
        "confidence_radius"
        if diagnostics["converged"]
        else "maximum_fit_budget"
    )
    if moments.mean is None:
        raise RuntimeError("reference path received no independent fits")
    return moments.mean.detach().cpu(), diagnostics


def _build_adaptive_reference_optimum_path(
    initial_model: nn.Module,
    dataset: Dataset,
    train_targets: Tensor,
    partitions: DatasetPartitions,
    p_values: Sequence[float],
    config: ExperimentConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ReferenceOptimumPath:
    reference = config.reference
    models: list[nn.Module] = []
    parameter_paths: list[list[Tensor]] = []
    plan_paths: list[list[ReferenceSamplePlan]] = []
    fit_rows: list[dict[str, Any]] = []
    decision_counts: list[int] = []
    decision_converged: list[bool] = []

    def add_fit_through(final_step: int) -> None:
        fit_index = len(models)
        model = copy.deepcopy(initial_model).to(device=device, dtype=dtype)
        fit_parameters: list[Tensor] = []
        fit_plans: list[ReferenceSamplePlan] = []
        for step in range(final_step + 1):
            parameter, row, plan = _fit_reference_point(
                model,
                dataset,
                train_targets,
                partitions,
                config,
                step=step,
                p_value=float(p_values[step]),
                fit_index=fit_index,
                device=device,
                dtype=dtype,
            )
            fit_parameters.append(parameter)
            fit_plans.append(plan)
            fit_rows.append(row)
        models.append(model)
        parameter_paths.append(fit_parameters)
        plan_paths.append(fit_plans)

    for _ in range(reference.calibration_min_fits):
        add_fit_through(0)

    progress = tqdm(
        range(len(p_values)),
        desc="reference-optimum path",
        unit="point",
        leave=False,
    )
    for step in progress:
        if step > 0:
            for fit_index, model in enumerate(models):
                parameter, row, plan = _fit_reference_point(
                    model,
                    dataset,
                    train_targets,
                    partitions,
                    config,
                    step=step,
                    p_value=float(p_values[step]),
                    fit_index=fit_index,
                    device=device,
                    dtype=dtype,
                )
                parameter_paths[fit_index].append(parameter)
                plan_paths[fit_index].append(plan)
                fit_rows.append(row)

        observations = (
            [path[0] for path in parameter_paths]
            if step == 0
            else [path[step] - path[step - 1] for path in parameter_paths]
        )
        _, diagnostics = _mean_diagnostics(observations, config)
        while (
            not diagnostics["converged"]
            and len(models) < reference.calibration_max_fits
        ):
            add_fit_through(step)
            observation = (
                parameter_paths[-1][0]
                if step == 0
                else parameter_paths[-1][step]
                - parameter_paths[-1][step - 1]
            )
            observations.append(observation)
            _, diagnostics = _mean_diagnostics(observations, config)
            progress.set_postfix(fits=len(models), refresh=False)
        decision_counts.append(len(models))
        decision_converged.append(bool(diagnostics["converged"]))

    replicate_parameters = torch.stack(
        [torch.stack(path) for path in parameter_paths]
    )
    parameters = replicate_parameters.mean(dim=0)
    displacements = parameters[1:] - parameters[:-1]
    displacement_diagnostics: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for step, p_value in enumerate(p_values):
        observations = (
            list(replicate_parameters[:, 0])
            if step == 0
            else list(
                replicate_parameters[:, step]
                - replicate_parameters[:, step - 1]
            )
        )
        _, diagnostics = _mean_diagnostics(observations, config)
        diagnostics.update(
            {
                "step": step,
                "p": float(p_value),
                "estimand": (
                    "initial_reference_parameter"
                    if step == 0
                    else "adjacent_reference_displacement"
                ),
                "fit_count_at_decision": decision_counts[step],
                "converged_at_decision": decision_converged[step],
                "final_fit_count": len(models),
            }
        )
        displacement_diagnostics.append(diagnostics)
        point_fits = [row for row in fit_rows if row["step"] == step]
        score_rows = [row["validation_score"] for row in point_fits]
        rows.append(
            {
                "step": step,
                "p": float(p_value),
                "fit_count": len(point_fits),
                "parameter_norm": float(
                    torch.linalg.vector_norm(parameters[step])
                ),
                "mean_validation_score_norm": sum(
                    row["mean_norm"] for row in score_rows
                )
                / len(score_rows),
                "max_validation_stationarity_upper_bound": max(
                    row["stationarity_upper_bound"] for row in score_rows
                ),
                "zero_in_six_sigma_ball_fraction": sum(
                    bool(row["zero_in_confidence_ball"])
                    for row in score_rows
                )
                / len(score_rows),
                "mean_training_validation_unique_overlap_fraction": sum(
                    row["training_validation_unique_overlap_fraction"]
                    for row in point_fits
                )
                / len(point_fits),
                "displacement_convergence": diagnostics,
            }
        )

    canonical_plans = tuple(plan_paths[0])
    path = ReferenceOptimumPath(
        p_values=tuple(float(p) for p in p_values),
        parameters=parameters,
        displacements=displacements,
        rows=tuple(rows),
        sample_plans=canonical_plans,
        content_hash=_path_hash(
            p_values,
            parameters,
            canonical_plans,
            replicate_parameters=replicate_parameters,
        ),
        replicate_parameters=replicate_parameters,
        fit_rows=tuple(fit_rows),
        displacement_diagnostics=tuple(displacement_diagnostics),
    )
    path.validate()
    return path


def build_reference_optimum_path(
    initial_model: nn.Module,
    dataset: Dataset,
    train_targets: Tensor,
    partitions: DatasetPartitions,
    p_values: Sequence[float],
    config: ExperimentConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> ReferenceOptimumPath:
    """Build the versioned warm-started reference-optimum path."""

    if not p_values:
        raise ValueError("p_values cannot be empty")
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("p_values must lie in [0, 1]")
    if config.schema_version < 8:
        return _build_fixed_reference_optimum_path(
            initial_model,
            dataset,
            train_targets,
            partitions,
            p_values,
            config,
            device=device,
            dtype=dtype,
        )
    return _build_adaptive_reference_optimum_path(
        initial_model,
        dataset,
        train_targets,
        partitions,
        p_values,
        config,
        device=device,
        dtype=dtype,
    )
