"""Chunked reference Fisher estimation, caching, and paired stencils."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from .convergence import HilbertMean
from .derivatives import LossFunction, per_sample_derivatives
from .fisher import LFUBatchEstimate
from .initialization import state_dict_hash
from .mnist_data import ReferenceSamplePlan
from .parameters import ParameterLayout

REFERENCE_FISHER_SCHEMA_VERSION = 1
REFERENCE_CACHE_SCHEMA_VERSION = 1

PerSampleLossFunction = Callable[[Tensor, Tensor], Tensor]


class ReferenceError(RuntimeError):
    """Raised when a reference estimate or cache artifact is invalid."""


@dataclasses.dataclass(frozen=True)
class ReferenceFisherEstimate:
    matrix: Tensor
    sample_count: int
    chunk_size: int
    elapsed_seconds: float
    score_gradient_count: int
    weight_mean: float
    weight_min: float
    weight_max: float
    effective_sample_size: float
    importance_weighted: bool
    convergence: dict[str, Any] | None = None
    dependence: dict[str, Any] | None = None

    def diagnostics_mapping(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "chunk_size": self.chunk_size,
            "elapsed_seconds": self.elapsed_seconds,
            "score_gradient_count": self.score_gradient_count,
            "weight_mean": self.weight_mean,
            "weight_min": self.weight_min,
            "weight_max": self.weight_max,
            "effective_sample_size": self.effective_sample_size,
            "importance_weighted": self.importance_weighted,
            "convergence": self.convergence,
            "dependence": self.dependence,
        }


@dataclasses.dataclass(frozen=True)
class ReferenceLFUEstimate:
    estimate: LFUBatchEstimate
    sample_count: int
    chunk_size: int
    elapsed_seconds: float
    score_gradient_count: int
    hvp_count: int


@dataclasses.dataclass(frozen=True)
class CentralStencilResult:
    derivative: Tensor
    plus: ReferenceFisherEstimate
    minus: ReferenceFisherEstimate
    epsilon: float
    direction_norm: float
    importance_weighted: bool
    plus_cache_digest: str
    minus_cache_digest: str


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_reference_inputs(
    model: nn.Module,
    layout: ParameterLayout,
    plan: ReferenceSamplePlan,
    chunk_size: int,
) -> None:
    layout.validate_module(model)
    plan.validate()
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise ValueError("chunk_size must be an integer")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")


def _reference_loader(
    dataset: Dataset,
    plan: ReferenceSamplePlan,
    *,
    chunk_size: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, plan.observation_indices),
        batch_size=chunk_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def chunked_reference_fisher(
    model: nn.Module,
    dataset: Dataset,
    plan: ReferenceSamplePlan,
    loss_function: LossFunction,
    layout: ParameterLayout,
    *,
    chunk_size: int,
    device: torch.device,
    derivative_dtype: torch.dtype,
    matrix_dtype: torch.dtype = torch.float64,
    strategy: str = "vmap",
    num_workers: int = 0,
    sampling_model: nn.Module | None = None,
    per_sample_loss_function: PerSampleLossFunction | None = None,
) -> ReferenceFisherEstimate:
    """Estimate a Fisher using scores and optional likelihood-ratio weights.

    When ``sampling_model`` is supplied, observations are treated as draws from
    that model and weighted by ``f_model(x) / f_sampling(x)``. This permits a
    paired estimate at a perturbed parameter without dropping the derivative of
    the probability measure.
    """

    _validate_reference_inputs(model, layout, plan, chunk_size)
    if sampling_model is not None:
        layout.validate_module(sampling_model)
        if per_sample_loss_function is None:
            raise ValueError(
                "per_sample_loss_function is required for importance weighting"
            )
    model.eval()
    if sampling_model is not None:
        sampling_model.eval()

    accumulator = torch.zeros(
        layout.total_numel,
        layout.total_numel,
        dtype=matrix_dtype,
        device=device,
    )
    weight_sum = torch.zeros((), dtype=matrix_dtype, device=device)
    weight_square_sum = torch.zeros((), dtype=matrix_dtype, device=device)
    weight_min = torch.full((), torch.inf, dtype=matrix_dtype, device=device)
    weight_max = torch.zeros((), dtype=matrix_dtype, device=device)
    sample_count = 0
    loader = _reference_loader(
        dataset,
        plan,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    _synchronize(device)
    start = time.perf_counter()
    for inputs, targets in loader:
        inputs = inputs.to(device=device, dtype=derivative_dtype)
        targets = targets.to(device=device)
        derivatives = per_sample_derivatives(
            model,
            inputs,
            targets,
            loss_function,
            layout,
            strategy=strategy,
        )
        gradients = derivatives.gradients.to(dtype=matrix_dtype)

        if sampling_model is None:
            weights = torch.ones(
                gradients.shape[0],
                dtype=matrix_dtype,
                device=device,
            )
        else:
            with torch.no_grad():
                target_losses = per_sample_loss_function(
                    model(inputs),
                    targets,
                ).to(dtype=matrix_dtype)
                sampling_losses = per_sample_loss_function(
                    sampling_model(inputs),
                    targets,
                ).to(dtype=matrix_dtype)
                log_weights = sampling_losses - target_losses
                weights = torch.exp(log_weights)
            if not torch.isfinite(weights).all():
                raise ReferenceError("importance weights are nonfinite")

        accumulator.add_(gradients.mT @ (weights.unsqueeze(1) * gradients))
        weight_sum.add_(weights.sum())
        weight_square_sum.add_(weights.square().sum())
        weight_min.copy_(torch.minimum(weight_min, weights.min()))
        weight_max.copy_(torch.maximum(weight_max, weights.max()))
        sample_count += gradients.shape[0]

    _synchronize(device)
    elapsed = time.perf_counter() - start
    if sample_count != plan.sample_size:
        raise ReferenceError(
            f"processed {sample_count} scores for a {plan.sample_size}-sample plan"
        )
    effective_sample_size = weight_sum.square() / weight_square_sum
    matrix = (accumulator / sample_count).detach().cpu()
    return ReferenceFisherEstimate(
        matrix=matrix,
        sample_count=sample_count,
        chunk_size=chunk_size,
        elapsed_seconds=elapsed,
        score_gradient_count=sample_count,
        weight_mean=float(weight_sum / sample_count),
        weight_min=float(weight_min),
        weight_max=float(weight_max),
        effective_sample_size=float(effective_sample_size),
        importance_weighted=sampling_model is not None,
    )


def adaptive_reference_fisher(
    model: nn.Module,
    dataset: Dataset,
    plan: ReferenceSamplePlan,
    loss_function: LossFunction,
    layout: ParameterLayout,
    *,
    chunk_size: int,
    minimum_chunks: int,
    sigma: float,
    relative_epsilon: float,
    absolute_epsilon: float,
    device: torch.device,
    derivative_dtype: torch.dtype,
    matrix_dtype: torch.dtype = torch.float64,
    strategy: str = "vmap",
    num_workers: int = 0,
) -> ReferenceFisherEstimate:
    """Estimate a Fisher until its Frobenius confidence radius is small.

    Each independently sampled chunk contributes one matrix-valued observation.
    Welford's scalar Hilbert-space moment therefore estimates the sum of all
    entry-wise variances without materializing a covariance of Fisher entries.
    """

    _validate_reference_inputs(model, layout, plan, chunk_size)
    if not isinstance(minimum_chunks, int) or minimum_chunks < 2:
        raise ValueError("minimum_chunks must be an integer >= 2")
    if plan.sample_size % chunk_size:
        raise ValueError("adaptive Fisher plans must contain full equal chunks")
    maximum_chunks = plan.sample_size // chunk_size
    if minimum_chunks > maximum_chunks:
        raise ValueError("minimum_chunks exceeds the Fisher sample budget")
    for name, value in {
        "sigma": sigma,
        "relative_epsilon": relative_epsilon,
        "absolute_epsilon": absolute_epsilon,
    }.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be positive and finite")

    model.eval()
    moments = HilbertMean()
    chunk_index_sets: list[set[int]] = []
    sample_count = 0
    loader = _reference_loader(
        dataset,
        plan,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    _synchronize(device)
    start = time.perf_counter()
    for chunk_index, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device=device, dtype=derivative_dtype)
        targets = targets.to(device=device)
        derivatives = per_sample_derivatives(
            model,
            inputs,
            targets,
            loss_function,
            layout,
            strategy=strategy,
        )
        gradients = derivatives.gradients.to(dtype=matrix_dtype)
        chunk_fisher = gradients.mT @ gradients / gradients.shape[0]
        moments.update(chunk_fisher)
        start_index = chunk_index * chunk_size
        chunk_index_sets.append(
            set(plan.observation_indices[start_index : start_index + chunk_size])
        )
        sample_count += gradients.shape[0]
        diagnostics = moments.diagnostics(
            sigma=sigma,
            relative_epsilon=relative_epsilon,
            absolute_epsilon=absolute_epsilon,
            minimum_count=minimum_chunks,
            maximum_count=maximum_chunks,
        )
        if diagnostics["converged"]:
            break

    _synchronize(device)
    elapsed = time.perf_counter() - start
    diagnostics = moments.diagnostics(
        sigma=sigma,
        relative_epsilon=relative_epsilon,
        absolute_epsilon=absolute_epsilon,
        minimum_count=minimum_chunks,
        maximum_count=maximum_chunks,
    )
    diagnostics["stopping_reason"] = (
        "confidence_radius" if diagnostics["converged"] else "maximum_budget"
    )
    diagnostics["geometry"] = "frobenius"
    diagnostics["sample_count"] = sample_count

    used_indices = plan.observation_indices[:sample_count]
    adjacent_overlaps = [
        len(left & right) / max(min(len(left), len(right)), 1)
        for left, right in zip(
            chunk_index_sets[:-1], chunk_index_sets[1:], strict=True
        )
    ]
    dependence = {
        "draw_duplicate_fraction": 1.0 - len(set(used_indices)) / sample_count,
        "mean_adjacent_chunk_index_overlap": (
            None
            if not adjacent_overlaps
            else sum(adjacent_overlaps) / len(adjacent_overlaps)
        ),
        "lag_one_frobenius_correlation": diagnostics[
            "lag_one_hilbert_correlation"
        ],
    }
    if moments.mean is None:
        raise ReferenceError("adaptive Fisher received no chunks")
    return ReferenceFisherEstimate(
        matrix=moments.mean.detach().cpu(),
        sample_count=sample_count,
        chunk_size=chunk_size,
        elapsed_seconds=elapsed,
        score_gradient_count=sample_count,
        weight_mean=1.0,
        weight_min=1.0,
        weight_max=1.0,
        effective_sample_size=float(sample_count),
        importance_weighted=False,
        convergence=diagnostics,
        dependence=dependence,
    )


def chunked_lfu_estimate(
    model: nn.Module,
    dataset: Dataset,
    plan: ReferenceSamplePlan,
    loss_function: LossFunction,
    layout: ParameterLayout,
    direction: Tensor,
    *,
    chunk_size: int,
    device: torch.device,
    derivative_dtype: torch.dtype,
    matrix_dtype: torch.dtype = torch.float64,
    strategy: str = "vmap",
    num_workers: int = 0,
) -> ReferenceLFUEstimate:
    """Estimate Fisher, AC, and residual matrices without retaining all scores."""

    _validate_reference_inputs(model, layout, plan, chunk_size)
    if direction.ndim != 1 or direction.numel() != layout.total_numel:
        raise ValueError("direction does not match the parameter layout")
    model_parameter = next(model.parameters())
    if (
        direction.device != model_parameter.device
        or direction.dtype != derivative_dtype
    ):
        raise ValueError("direction must use the derivative dtype and device")
    model.eval()

    shape = (layout.total_numel, layout.total_numel)
    fisher_sum = torch.zeros(shape, dtype=matrix_dtype, device=device)
    ac_sum = torch.zeros_like(fisher_sum)
    residual_sum = torch.zeros_like(fisher_sum)
    matrix_direction = direction.to(dtype=matrix_dtype)
    sample_count = 0
    loader = _reference_loader(
        dataset,
        plan,
        chunk_size=chunk_size,
        num_workers=num_workers,
    )

    _synchronize(device)
    start = time.perf_counter()
    for inputs, targets in loader:
        inputs = inputs.to(device=device, dtype=derivative_dtype)
        targets = targets.to(device=device)
        derivatives = per_sample_derivatives(
            model,
            inputs,
            targets,
            loss_function,
            layout,
            direction=direction,
            strategy=strategy,
        )
        gradients = derivatives.gradients.to(dtype=matrix_dtype)
        hvps = derivatives.hvps.to(dtype=matrix_dtype)
        directional_gradients = gradients @ matrix_direction
        fisher_sum.add_(gradients.mT @ gradients)
        ac_sum.add_(
            -gradients.mT
            @ (directional_gradients.unsqueeze(1) * gradients)
        )
        residual_sum.add_(
            hvps.mT @ gradients + gradients.mT @ hvps
        )
        sample_count += gradients.shape[0]

    _synchronize(device)
    elapsed = time.perf_counter() - start
    if sample_count != plan.sample_size:
        raise ReferenceError("LFU sample count does not match reference plan")
    estimate = LFUBatchEstimate(
        fisher=(fisher_sum / sample_count).detach().cpu(),
        amari_chentsov=(ac_sum / sample_count).detach().cpu(),
        residual=(residual_sum / sample_count).detach().cpu(),
    )
    return ReferenceLFUEstimate(
        estimate=estimate,
        sample_count=sample_count,
        chunk_size=chunk_size,
        elapsed_seconds=elapsed,
        score_gradient_count=sample_count,
        hvp_count=sample_count,
    )


def central_fisher_stencil(
    model: nn.Module,
    dataset: Dataset,
    plan: ReferenceSamplePlan,
    loss_function: LossFunction,
    per_sample_loss_function: PerSampleLossFunction,
    layout: ParameterLayout,
    direction: Tensor,
    epsilon: float,
    *,
    chunk_size: int,
    device: torch.device,
    derivative_dtype: torch.dtype,
    matrix_dtype: torch.dtype = torch.float64,
    strategy: str = "vmap",
    num_workers: int = 0,
    importance_weighted: bool = True,
    cache_store: "ReferenceFisherStore | None" = None,
) -> CentralStencilResult:
    """Calculate a paired central stencil around ``model`` along ``direction``."""

    if not isinstance(epsilon, (int, float)) or epsilon <= 0:
        raise ValueError("epsilon must be positive")
    layout.validate_module(model)
    if direction.ndim != 1 or direction.numel() != layout.total_numel:
        raise ValueError("direction does not match the parameter layout")
    direction_norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(direction_norm) or float(direction_norm) == 0.0:
        raise ValueError("direction must have a finite nonzero norm")

    base_vector = layout.flatten_module(model, detach=True)
    plus_model = copy.deepcopy(model)
    minus_model = copy.deepcopy(model)
    plus_layout = ParameterLayout.from_module(plus_model)
    minus_layout = ParameterLayout.from_module(minus_model)
    plus_layout.assert_metadata(layout.metadata())
    minus_layout.assert_metadata(layout.metadata())
    plus_layout.copy_vector_to_module(
        plus_model,
        base_vector + float(epsilon) * direction,
    )
    minus_layout.copy_vector_to_module(
        minus_model,
        base_vector - float(epsilon) * direction,
    )
    sampling_model = model if importance_weighted else None

    cache_digests: list[str] = []

    def estimate(
        target_model: nn.Module,
        target_layout: ParameterLayout,
    ) -> ReferenceFisherEstimate:
        key = reference_cache_key(
            target_model,
            target_layout,
            plan,
            derivative_dtype=derivative_dtype,
            matrix_dtype=matrix_dtype,
            sampling_model=sampling_model,
        )
        cache_digests.append(key.digest)
        if cache_store is not None and cache_store.exists(key):
            return cache_store.load(key, target_layout)
        result = chunked_reference_fisher(
            target_model,
            dataset,
            plan,
            loss_function,
            target_layout,
            chunk_size=chunk_size,
            device=device,
            derivative_dtype=derivative_dtype,
            matrix_dtype=matrix_dtype,
            strategy=strategy,
            num_workers=num_workers,
            sampling_model=sampling_model,
            per_sample_loss_function=per_sample_loss_function,
        )
        if cache_store is not None:
            cache_store.save(key, result)
        return result

    plus = estimate(plus_model, plus_layout)
    minus = estimate(minus_model, minus_layout)
    derivative = (plus.matrix - minus.matrix) / (2 * float(epsilon))
    return CentralStencilResult(
        derivative=derivative,
        plus=plus,
        minus=minus,
        epsilon=float(epsilon),
        direction_norm=float(direction_norm),
        importance_weighted=importance_weighted,
        plus_cache_digest=cache_digests[0],
        minus_cache_digest=cache_digests[1],
    )


def relative_frobenius_error(estimate: Tensor, target: Tensor) -> float:
    if estimate.shape != target.shape:
        raise ValueError("matrices must have equal shapes")
    denominator = torch.linalg.matrix_norm(target, ord="fro")
    floor = torch.finfo(target.dtype).eps
    return float(
        torch.linalg.matrix_norm(estimate - target, ord="fro")
        / denominator.clamp_min(floor)
    )


def convergence_diagnostics(
    estimates: dict[int, Tensor],
) -> list[dict[str, float | int | None]]:
    if not estimates:
        raise ValueError("at least one estimate is required")
    sizes = sorted(estimates)
    if sizes != list(estimates):
        estimates = {size: estimates[size] for size in sizes}
    largest = estimates[sizes[-1]]
    diagnostics = []
    previous = None
    for size in sizes:
        current = estimates[size]
        diagnostics.append(
            {
                "sample_size": size,
                "relative_to_largest": relative_frobenius_error(
                    current,
                    largest,
                ),
                "relative_to_previous": (
                    None
                    if previous is None
                    else relative_frobenius_error(current, previous)
                ),
            }
        )
        previous = current
    return diagnostics


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclasses.dataclass(frozen=True)
class ReferenceCacheKey:
    target_checkpoint_hash: str
    sampling_checkpoint_hash: str | None
    parameter_layout: dict[str, Any]
    reference_plan_hash: str
    p: float
    sample_count: int
    derivative_dtype: str
    matrix_dtype: str
    importance_weighted: bool
    schema_version: int = REFERENCE_CACHE_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            _canonical_json(self.to_mapping()).encode("utf-8")
        ).hexdigest()


def reference_cache_key(
    model: nn.Module,
    layout: ParameterLayout,
    plan: ReferenceSamplePlan,
    *,
    derivative_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
    sampling_model: nn.Module | None = None,
) -> ReferenceCacheKey:
    layout.validate_module(model)
    plan.validate()
    if sampling_model is not None:
        layout.validate_module(sampling_model)
    return ReferenceCacheKey(
        target_checkpoint_hash=state_dict_hash(model.state_dict()),
        sampling_checkpoint_hash=(
            None
            if sampling_model is None
            else state_dict_hash(sampling_model.state_dict())
        ),
        parameter_layout=layout.metadata(),
        reference_plan_hash=plan.content_hash,
        p=plan.p,
        sample_count=plan.sample_size,
        derivative_dtype=str(derivative_dtype),
        matrix_dtype=str(matrix_dtype),
        importance_weighted=sampling_model is not None,
    )


def _matrix_hash(matrix: Tensor) -> str:
    cpu = matrix.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(str(tuple(cpu.shape)).encode("ascii"))
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ReferenceFisherStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path_for(self, key: ReferenceCacheKey) -> Path:
        return self.root / key.digest

    def exists(self, key: ReferenceCacheKey) -> bool:
        return (self.path_for(key) / "COMPLETED").is_file()

    def save(
        self,
        key: ReferenceCacheKey,
        estimate: ReferenceFisherEstimate,
    ) -> Path:
        destination = self.path_for(key)
        if destination.exists():
            raise ReferenceError(f"reference cache entry already exists: {destination}")
        if estimate.sample_count != key.sample_count:
            raise ReferenceError("estimate sample count does not match cache key")
        expected_shape = (
            key.parameter_layout["total_numel"],
            key.parameter_layout["total_numel"],
        )
        if tuple(estimate.matrix.shape) != expected_shape:
            raise ReferenceError("reference matrix shape does not match cache key")

        incomplete_root = self.root / ".incomplete"
        incomplete_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f"{key.digest}.", dir=incomplete_root)
        )
        try:
            matrix = estimate.matrix.detach().cpu().contiguous()
            with (temporary / "fisher.pt").open("wb") as stream:
                torch.save(matrix, stream)
                stream.flush()
                os.fsync(stream.fileno())
            metadata = {
                "schema_version": REFERENCE_FISHER_SCHEMA_VERSION,
                "key": key.to_mapping(),
                "key_digest": key.digest,
                "matrix_hash": _matrix_hash(matrix),
                "diagnostics": estimate.diagnostics_mapping(),
                "status": "completed",
            }
            with (temporary / "metadata.json").open("wb") as stream:
                stream.write((_canonical_json(metadata) + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            with (temporary / "COMPLETED").open("wb") as marker:
                marker.flush()
                os.fsync(marker.fileno())
            self.root.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination

    def load(
        self,
        key: ReferenceCacheKey,
        layout: ParameterLayout,
    ) -> ReferenceFisherEstimate:
        path = self.path_for(key)
        if not (path / "COMPLETED").is_file():
            raise ReferenceError(f"reference cache entry is incomplete: {path}")
        try:
            metadata = json.loads(
                (path / "metadata.json").read_text(encoding="utf-8")
            )
            matrix = torch.load(
                path / "fisher.pt",
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, json.JSONDecodeError, RuntimeError) as exc:
            raise ReferenceError(f"could not load reference cache: {exc}") from exc
        if metadata.get("schema_version") != REFERENCE_FISHER_SCHEMA_VERSION:
            raise ReferenceError("unsupported reference Fisher schema")
        if metadata.get("key") != key.to_mapping():
            raise ReferenceError("reference cache key does not match metadata")
        try:
            layout.assert_metadata(key.parameter_layout)
        except ValueError as exc:
            raise ReferenceError("reference parameter layout mismatch") from exc
        if _matrix_hash(matrix) != metadata.get("matrix_hash"):
            raise ReferenceError("reference matrix hash does not match metadata")
        expected_shape = (layout.total_numel, layout.total_numel)
        if tuple(matrix.shape) != expected_shape:
            raise ReferenceError("cached reference matrix shape is incompatible")
        diagnostics = metadata["diagnostics"]
        return ReferenceFisherEstimate(
            matrix=matrix,
            sample_count=diagnostics["sample_count"],
            chunk_size=diagnostics["chunk_size"],
            elapsed_seconds=diagnostics["elapsed_seconds"],
            score_gradient_count=diagnostics["score_gradient_count"],
            weight_mean=diagnostics["weight_mean"],
            weight_min=diagnostics["weight_min"],
            weight_max=diagnostics["weight_max"],
            effective_sample_size=diagnostics["effective_sample_size"],
            importance_weighted=diagnostics["importance_weighted"],
            convergence=diagnostics.get("convergence"),
            dependence=diagnostics.get("dependence"),
        )
