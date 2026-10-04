"""Conditional bias-variance response study at frozen Plan 12 anchors."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.derivatives import per_sample_derivatives
from src.ewc import build_optimizer, mixture_ewc_strength, take_ewc_proposal
from src.mnist_model import build_canonical_model, mnist_nll
from src.parameters import ParameterLayout
from src.representations import DenseFisher, LowRankDiagonalFisher, representation_from_artifact

from ..phase4_metrics import evaluate_materialized_classifier
from ..run import CALIBRATION_BINS, _learner_optimizer_config
from .anchors import ANCHOR_REQUIRED, ensure_anchor_path, load_anchor
from .artifacts import UnitStore
from .assets import runtime
from .config import PI
from .gauge import build_gauge_fixed_model
from .ridge import RidgeFisher, fisher_scale, leading_subspace
from .trajectory import _resolved_basis


REFERENCE_REQUIRED = ("reference.pt", "summary.json")
TARGET_REQUIRED = ("target.pt", "summary.json")
BRANCH_REQUIRED = ("estimate.pt", "summary.json")


@dataclasses.dataclass(frozen=True)
class LocalCondition:
    name: str
    chart: bool = True
    current_only: bool = False
    no_update: bool = False
    fisher_kind: str = "compressed"
    ridge_geometry: str | None = None
    ridge_ratio: float = 0.0
    heuristic_top: bool = False

    def __post_init__(self) -> None:
        if self.fisher_kind not in {"compressed", "dense"}:
            raise ValueError("local Fisher kind must be compressed or dense")
        if self.ridge_geometry not in {None, "isotropic", "tail"}:
            raise ValueError("invalid local ridge geometry")
        if self.ridge_ratio < 0:
            raise ValueError("local ridge ratio must be nonnegative")
        if self.no_update and (self.current_only or self.ridge_geometry is not None):
            raise ValueError("no-update is a separate boundary condition")

    def mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _ratio_name(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def local_conditions(ratios: tuple[float, ...]) -> tuple[LocalCondition, ...]:
    nonzero = tuple(value for value in ratios if value > 0)
    return (
        LocalCondition("no_update", no_update=True),
        LocalCondition("current_only", current_only=True),
        LocalCondition("gauge_no_ridge"),
        LocalCondition("dense_archive_no_ridge", fisher_kind="dense"),
        *(LocalCondition(f"isotropic_{_ratio_name(value)}", ridge_geometry="isotropic", ridge_ratio=value) for value in nonzero),
        *(LocalCondition(f"tail_{_ratio_name(value)}", ridge_geometry="tail", ridge_ratio=value) for value in nonzero),
        LocalCondition("heuristic_top", ridge_geometry="isotropic", heuristic_top=True),
        LocalCondition("legacy_raw", chart=False),
    )


def _anchor_path(store: UnitStore, anchor_index: int, *, data_root: Path, resume: bool) -> Path:
    schedule = "linear" if anchor_index <= store.study.anchors_per_schedule else "sigmoid"
    return ensure_anchor_path(store, schedule, data_root=data_root, resume=resume)


def _load_model(
    anchor: dict[str, Any],
    condition: LocalCondition,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, ParameterLayout, Tensor, Tensor | LowRankDiagonalFisher]:
    if condition.chart:
        model, layout = build_gauge_fixed_model(seed, device=device, dtype=dtype)
        model.load_state_dict(anchor["chart_state"])
        dense = anchor["chart_dense_archive"].to(device=device)
        compressed = representation_from_artifact(anchor["chart_fisher"], device=device)
        parameter = anchor["chart_parameter"].to(device=device, dtype=dtype)
    else:
        model, layout = build_canonical_model(seed, device=device, dtype=dtype)
        model.load_state_dict(anchor["raw_state"])
        dense = anchor["raw_dense_archive"].to(device=device)
        compressed = representation_from_artifact(anchor["raw_fisher"], device=device)
        parameter = anchor["raw_parameter"].to(device=device, dtype=dtype)
    if not isinstance(compressed, LowRankDiagonalFisher):
        raise RuntimeError("local compressed Fisher is not rank-plus-diagonal")
    fisher: Tensor | LowRankDiagonalFisher = dense if condition.fisher_kind == "dense" else compressed
    return model, layout, parameter, fisher


def _operator(
    fisher: Tensor | LowRankDiagonalFisher,
    condition: LocalCondition,
) -> tuple[Tensor | LowRankDiagonalFisher | RidgeFisher, float, Tensor]:
    if isinstance(fisher, Tensor):
        eigenvalues, basis = leading_subspace(fisher, 8)
        top = float(eigenvalues[0])
    else:
        basis = _resolved_basis(fisher)
        top = float(torch.linalg.eigvalsh(fisher.to_dense())[-1])
    kappa = 0.01 * top if condition.heuristic_top else condition.ridge_ratio * fisher_scale(fisher)
    if condition.ridge_geometry is None:
        return fisher, 0.0, basis
    operator = RidgeFisher(
        DenseFisher(fisher) if isinstance(fisher, Tensor) else fisher,
        kappa,
        condition.ridge_geometry,
        basis if condition.ridge_geometry == "tail" else None,
    )
    return operator, kappa, basis


def _dense_reference_fisher(
    model: nn.Module,
    layout: ParameterLayout,
    inputs: Tensor,
    targets: Tensor,
    *,
    matrix_dtype: torch.dtype,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    accumulator = torch.zeros(layout.total_numel, layout.total_numel, device=inputs.device, dtype=matrix_dtype)
    gradients = []
    for start in range(0, inputs.shape[0], chunk_size):
        stop = min(start + chunk_size, inputs.shape[0])
        # Loss gradients equal negative scores; their outer products estimate the same Fisher.
        batch = per_sample_derivatives(
            model,
            inputs[start:stop],
            targets[start:stop],
            mnist_nll,
            layout,
            strategy="vmap",
        ).gradients.to(dtype=matrix_dtype)
        accumulator.add_(batch.mT @ batch)
        gradients.append(batch.cpu())
    matrix = accumulator / inputs.shape[0]
    return (matrix + matrix.mT) / 2, torch.cat(gradients)


def run_anchor_reference(
    store: UnitStore,
    anchor_index: int,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    anchor_path = _anchor_path(store, anchor_index, data_root=data_root, resume=True)
    unit = store.unit("phase1", "reference", anchor_index)
    session = store.begin(unit, REFERENCE_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, REFERENCE_REQUIRED)
        assert completed is not None
        return completed
    started = time.perf_counter()
    device, training_dtype, matrix_dtype = runtime(store.study)
    anchor = load_anchor(anchor_path, anchor_index)
    condition = LocalCondition("gauge_no_ridge")
    model, layout, _, _ = _load_model(
        anchor,
        condition,
        seed=store.study.seed("phase1:reference-model", anchor_index),
        device=device,
        dtype=training_dtype,
    )
    inputs = anchor["samples"]["population_inputs"].to(device=device, dtype=training_dtype)
    targets = anchor["samples"]["population_targets"].to(device=device)
    fisher, gradients = _dense_reference_fisher(
        model,
        layout,
        inputs,
        targets,
        matrix_dtype=matrix_dtype,
        chunk_size=store.study.protocol.fisher.chunk_size,
    )
    eigenvalues, basis = leading_subspace(fisher, 8)
    summary = {
        "anchor_index": anchor_index,
        "sample_count": inputs.shape[0],
        "trace": float(torch.trace(fisher)),
        "top_eigenvalues": eigenvalues[:16].tolist(),
        "minimum_eigenvalue": float(eigenvalues[-1]),
        "score_mean_norm": float(torch.linalg.vector_norm(gradients.mean(dim=0))),
        "wall_time_seconds": time.perf_counter() - started,
    }
    session.write_torch(
        "reference.pt",
        {"fisher": fisher.cpu(), "scores": gradients, "resolved_basis": basis.cpu()},
    )
    session.write_json("summary.json", summary)
    return store.finish(session, REFERENCE_REQUIRED)


def _evaluate(model: nn.Module, anchor: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> tuple[dict[str, float], Tensor]:
    samples = anchor["samples"]
    targets = samples["evaluation_targets"]
    panels = {
        "current": samples["current_evaluation_inputs"],
        "retention_000": samples["retention_000_inputs"],
        "retention_015": samples["retention_015_inputs"],
        "retention_030": samples["retention_030_inputs"],
    }
    metrics: dict[str, float] = {}
    for name, inputs in panels.items():
        values = evaluate_materialized_classifier(
            model,
            inputs,
            targets,
            batch_size=store_batch_size(inputs.shape[0]),
            device=device,
            dtype=dtype,
            calibration_bins=CALIBRATION_BINS,
            nine_prevalence=anchor["samples"]["nine_prevalence"],
        )
        metrics.update({f"{name}_{key}": value for key, value in values.items()})
    probe_inputs = panels["current"][: min(256, panels["current"].shape[0])].to(device=device, dtype=dtype)
    with torch.no_grad():
        logits = model(probe_inputs).detach().cpu()
    return metrics, logits


def store_batch_size(size: int) -> int:
    return min(256, size)


def _fit_once(
    store: UnitStore,
    anchor: dict[str, Any],
    condition: LocalCondition,
    inputs: Tensor,
    targets: Tensor,
    *,
    perturbation: Tensor | None = None,
) -> tuple[nn.Module, ParameterLayout, dict[str, Any], float, Tensor]:
    device, training_dtype, _ = runtime(store.study)
    model, layout, anchor_parameter, fisher = _load_model(
        anchor,
        condition,
        seed=store.study.seed("phase1:fit-model", anchor["anchor_index"]),
        device=device,
        dtype=training_dtype,
    )
    if perturbation is not None:
        layout.copy_vector_to_module(model, anchor_parameter + perturbation.to(anchor_parameter))
    operator, kappa, basis = _operator(fisher, condition)
    if condition.no_update:
        return model, layout, {"stopping_reason": "no_update", "final_gradient_norm": None}, kappa, basis
    optimizer = build_optimizer(model, _learner_optimizer_config(store.study.protocol))
    result = take_ewc_proposal(
        model,
        layout,
        inputs.to(device=device, dtype=training_dtype),
        targets.to(device=device),
        operator.to(dtype=training_dtype),
        _learner_optimizer_config(store.study.protocol),
        optimizer,
        adaptation_weight=1.0 if condition.current_only else PI,
        penalty_anchor=anchor_parameter,
    )
    return model, layout, result.metrics_mapping(), kappa, basis


def run_local_target(
    store: UnitStore,
    anchor_index: int,
    condition: LocalCondition,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    anchor_path = _anchor_path(store, anchor_index, data_root=data_root, resume=True)
    unit = store.unit("phase1", "target", anchor_index, condition=condition.name, detail=condition.mapping())
    session = store.begin(unit, TARGET_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, TARGET_REQUIRED)
        assert completed is not None
        return completed
    started = time.perf_counter()
    anchor = load_anchor(anchor_path, anchor_index)
    inputs = anchor["samples"]["population_inputs"]
    targets = anchor["samples"]["population_targets"]
    model, layout, health, kappa, basis = _fit_once(store, anchor, condition, inputs, targets)
    vector = layout.flatten_module(model, detach=True).cpu()
    if condition.no_update:
        # There is no fitted target to reproduce for the boundary condition.
        repeated_health = health
        repeated_vector = vector.clone()
    else:
        generator = torch.Generator().manual_seed(store.study.seed("phase1:target-repeat", anchor_index))
        perturbation = torch.randn(vector.shape, generator=generator, dtype=vector.dtype)
        perturbation *= 1e-6 / torch.linalg.vector_norm(perturbation).clamp_min(torch.finfo(vector.dtype).tiny)
        repeated_model, repeated_layout, repeated_health, _, _ = _fit_once(
            store,
            anchor,
            condition,
            inputs,
            targets,
            perturbation=perturbation,
        )
        repeated_vector = repeated_layout.flatten_module(repeated_model, detach=True).cpu()
    device, training_dtype, _ = runtime(store.study)
    metrics, logits = _evaluate(model, anchor, device=device, dtype=training_dtype)
    summary = {
        "anchor_index": anchor_index,
        "condition": condition.mapping(),
        "kappa": kappa,
        "beta": mixture_ewc_strength(PI),
        "tau": mixture_ewc_strength(PI) * kappa,
        "resolved_rank": basis.shape[1],
        "fit_health": health,
        "repeat_fit_health": repeated_health,
        "repeat_parameter_distance": float(torch.linalg.vector_norm(vector - repeated_vector)),
        "metrics": metrics,
        "wall_time_seconds": time.perf_counter() - started,
    }
    session.write_torch(
        "target.pt",
        {
            "parameter": vector,
            "repeat_parameter": repeated_vector,
            "current_logits": logits,
            "current_probabilities": logits.softmax(dim=1),
        },
    )
    session.write_json("summary.json", summary)
    return store.finish(session, TARGET_REQUIRED)


def run_local_branch(
    store: UnitStore,
    anchor_index: int,
    batch_index: int,
    condition: LocalCondition,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    anchor_path = _anchor_path(store, anchor_index, data_root=data_root, resume=True)
    unit_index = (anchor_index - 1) * store.study.local_batches_per_anchor + batch_index
    unit = store.unit(
        "phase1",
        "branch",
        unit_index,
        condition=condition.name,
        detail={"anchor_index": anchor_index, "batch_index": batch_index, **condition.mapping()},
    )
    session = store.begin(unit, BRANCH_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, BRANCH_REQUIRED)
        assert completed is not None
        return completed
    started = time.perf_counter()
    anchor = load_anchor(anchor_path, anchor_index)
    inputs = anchor["samples"]["batch_inputs"][batch_index - 1]
    targets = anchor["samples"]["batch_targets"][batch_index - 1]
    model, layout, health, kappa, basis = _fit_once(store, anchor, condition, inputs, targets)
    vector = layout.flatten_module(model, detach=True).cpu()
    device, training_dtype, _ = runtime(store.study)
    metrics, logits = _evaluate(model, anchor, device=device, dtype=training_dtype)
    summary = {
        "anchor_index": anchor_index,
        "batch_index": batch_index,
        "condition": condition.mapping(),
        "kappa": kappa,
        "beta": mixture_ewc_strength(PI),
        "tau": mixture_ewc_strength(PI) * kappa,
        "resolved_rank": basis.shape[1],
        "fit_health": health,
        "metrics": metrics,
        "wall_time_seconds": time.perf_counter() - started,
    }
    session.write_torch(
        "estimate.pt",
        {"parameter": vector, "current_logits": logits, "current_probabilities": logits.softmax(dim=1)},
    )
    session.write_json("summary.json", summary)
    return store.finish(session, BRANCH_REQUIRED)
