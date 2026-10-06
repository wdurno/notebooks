"""Plan 3-compatible digit-9-mixture assets and evaluation for Plan 13."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.config import DataConfig, InitializationConfig
from src.derivatives import per_sample_derivatives
from src.initialization import fit_p0_initialization
from src.mnist_data import (
    dataset_targets,
    generate_mixture_stream,
    load_mnist_datasets,
    partition_mnist,
)
from src.mnist_model import mnist_nll
from src.representations import LowRankDiagonalFisher
from src.seeding import derive_component_seed

from mnist_experiment.rotated_mnist.plan12.gauge import build_gauge_fixed_model
from mnist_experiment.rotated_mnist.run import _state_dict_cpu
from mnist_experiment.rotated_mnist.transform import tensor_content_hash

from ..artifacts import UnitStore
from ..rotation import runtime


MIXTURE_ASSET_REQUIRED = ("assets.pt", "summary.json", "checks.json")
MIXTURE_PI = 0.05


def nine_ovr_metrics(predictions: Tensor, targets: Tensor, p: float) -> dict[str, float]:
    """Evaluate digit 9 as a binary one-vs-rest classification problem."""
    if predictions.ndim != 1 or targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("predictions and targets must be matching one-dimensional tensors")
    if not 0.0 <= p <= 1.0:
        raise ValueError("mixture probability p must be in [0, 1]")
    nine = targets == 9
    if not bool(nine.any()) or not bool((~nine).any()):
        raise ValueError("digit-9 OvR evaluation requires both target classes")
    predicted_nine = predictions == 9
    recall = float(predicted_nine[nine].to(torch.float64).mean())
    specificity = float((~predicted_nine[~nine]).to(torch.float64).mean())
    return {
        "nine_ovr_accuracy": p * recall + (1 - p) * specificity,
        "nine_ovr_balanced_accuracy": 0.5 * (recall + specificity),
        "nine_ovr_recall": recall,
        "nine_ovr_specificity": specificity,
        "nine_ovr_false_positive_rate": 1 - specificity,
    }


def mixture_data_config(smoke: bool) -> DataConfig:
    return DataConfig(
        num_p_steps=4 if smoke else 100,
        samples_per_step=2 if smoke else 8,
        non_nine_sampling="empirical",
        initialization_size=512 if smoke else 30_000,
        online_pool_size=512 if smoke else 12_000,
        reference_pool_size=512 if smoke else 12_000,
        evaluation_size=512 if smoke else 10_000,
    )


def mixture_initialization_config(store: UnitStore) -> InitializationConfig:
    source = store.study.protocol.initialization
    return InitializationConfig(
        optimizer=source.optimizer,
        learning_rate=source.learning_rate,
        weight_decay=source.weight_decay,
        batch_size=source.batch_size,
        max_epochs=1 if store.study.smoke else source.max_epochs,
        target_non_nine_accuracy=None if store.study.smoke else 0.80,
        num_workers=0 if store.study.smoke else store.study.protocol.runtime.num_workers,
    )


def mixture_asset_unit(store: UnitStore, index: int) -> dict[str, Any]:
    data = mixture_data_config(store.study.smoke)
    return store.unit(
        "phase5",
        "mixture_assets",
        index,
        environment="digit9_mixture",
        detail={
            "num_p_steps": data.num_p_steps,
            "samples_per_step": data.samples_per_step,
            "non_nine_sampling": data.non_nine_sampling,
            "initial_fisher_samples": 32 if store.study.smoke else 20_000,
            "fisher_rank": 8,
        },
    )


def _materialize(dataset: Any, indices: list[int] | tuple[int, ...]) -> tuple[Tensor, Tensor]:
    pairs = [dataset[int(index)] for index in indices]
    return torch.stack([item[0] for item in pairs]), torch.tensor(
        [int(item[1]) for item in pairs], dtype=torch.long
    )


def _initial_fisher(
    store: UnitStore,
    model: torch.nn.Module,
    layout: Any,
    train_dataset: Any,
    reference_indices: tuple[int, ...],
    *,
    index: int,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> tuple[LowRankDiagonalFisher, dict[str, Any]]:
    targets = dataset_targets(train_dataset)
    candidates = torch.tensor(reference_indices, dtype=torch.long)
    candidates = candidates[targets[candidates] != 9]
    sample_size = 32 if store.study.smoke else 20_000
    generator = torch.Generator().manual_seed(store.study.seed("phase5:initial_fisher", index))
    selected = candidates[
        torch.randint(candidates.numel(), (sample_size,), generator=generator)
    ]
    dense = torch.zeros(
        layout.total_numel,
        layout.total_numel,
        device=device,
        dtype=matrix_dtype,
    )
    chunk_size = 16 if store.study.smoke else 512
    started = time.perf_counter()
    for start in range(0, sample_size, chunk_size):
        inputs, labels = _materialize(train_dataset, selected[start : start + chunk_size].tolist())
        gradients = per_sample_derivatives(
            model,
            inputs.to(device=device, dtype=training_dtype),
            labels.to(device=device),
            mnist_nll,
            layout,
            strategy="vmap",
        ).gradients.to(dtype=matrix_dtype)
        dense.add_(gradients.mT @ gradients)
    dense.div_(sample_size)
    dense = (dense + dense.mT) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(dense)
    values = eigenvalues[-8:].clamp_min(0)
    vectors = eigenvectors[:, -8:]
    factor = vectors * values.sqrt().unsqueeze(0)
    residual = (torch.diagonal(dense) - factor.square().sum(dim=1)).clamp_min(0)
    fisher = LowRankDiagonalFisher(factor, residual)
    return fisher, {
        "sample_size": sample_size,
        "rank": fisher.rank,
        "trace": float(fisher.diagonal_vector().sum()),
        "wall_seconds": time.perf_counter() - started,
        "sampling": "with_replacement_from_independent_p0_reference_pool",
    }


def ensure_mixture_assets(
    store: UnitStore,
    index: int,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    unit = mixture_asset_unit(store, index)
    session = store.begin(unit, MIXTURE_ASSET_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, MIXTURE_ASSET_REQUIRED)
        assert completed is not None
        return completed
    device, training_dtype, matrix_dtype = runtime(store.study)
    data = mixture_data_config(store.study.smoke)
    initialization_config = mixture_initialization_config(store)
    replica_seed = store.study.seed("phase5:mixture_replica", index)
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=False)
    train_targets = dataset_targets(train_dataset)
    test_targets = dataset_targets(test_dataset)
    partitions = partition_mnist(
        train_targets,
        test_targets,
        data,
        replica_seed=replica_seed,
    )
    stream = generate_mixture_stream(
        train_targets,
        partitions,
        data,
        seed=derive_component_seed(replica_seed, "online_stream"),
    )
    model, layout = build_gauge_fixed_model(
        derive_component_seed(replica_seed, "model_initialization"),
        device=device,
        dtype=training_dtype,
    )
    initialization = fit_p0_initialization(
        model,
        train_dataset,
        test_dataset,
        partitions,
        initialization_config,
        loader_seed=derive_component_seed(replica_seed, "initialization_loader"),
        device=device,
        dtype=training_dtype,
    )
    fisher, fisher_summary = _initial_fisher(
        store,
        model,
        layout,
        train_dataset,
        partitions.reference,
        index=index,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
    )
    flat_indices = [value for row in stream.observation_indices for value in row]
    stream_inputs, stream_targets = _materialize(train_dataset, flat_indices)
    stream_inputs = stream_inputs.reshape(
        data.num_p_steps,
        data.samples_per_step,
        *stream_inputs.shape[1:],
    )
    stream_targets = stream_targets.reshape(data.num_p_steps, data.samples_per_step)
    evaluation_inputs, evaluation_targets = _materialize(test_dataset, partitions.evaluation)
    assets = {
        "model": _state_dict_cpu(model),
        "parameter_layout": layout.metadata(),
        "initial_fisher": fisher.artifact_mapping(),
        "partitions": partitions.to_mapping(),
        "stream_plan": stream.to_mapping(),
        "stream_inputs": stream_inputs,
        "stream_targets": stream_targets,
        "evaluation_inputs": evaluation_inputs,
        "evaluation_targets": evaluation_targets,
    }
    checks = {
        "parameter_count": layout.total_numel,
        "parameter_count_is_487": layout.total_numel == 487,
        "stream_labels_match": torch.equal(
            stream_targets,
            torch.tensor(stream.class_labels, dtype=torch.long),
        ),
        "p0_initialization_has_no_nines": not bool((train_targets[list(partitions.initialization)] == 9).any()),
        "fixed_non_nine_conditional": stream.non_nine_sampling == "empirical",
    }
    if not all(value for key, value in checks.items() if key != "parameter_count"):
        raise RuntimeError(f"mixture asset checks failed: {checks}")
    summary = {
        "phase": "phase5",
        "environment": "digit9_mixture",
        "replica_index": index,
        "replica_seed": replica_seed,
        "p_values": list(stream.p_values),
        "samples_per_step": stream.samples_per_step,
        "stream_hash": stream.content_hash,
        "initialization": dataclasses.asdict(initialization),
        "initial_fisher": fisher_summary,
        "model_state_hash": tensor_content_hash(layout.flatten_module(model, detach=True).cpu()),
    }
    session.write_torch("assets.pt", assets)
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    return store.finish(session, MIXTURE_ASSET_REQUIRED)


@torch.no_grad()
def evaluate_mixture(
    model: torch.nn.Module,
    inputs: Tensor,
    targets: Tensor,
    p: float,
    *,
    chunk_size: int = 1024,
) -> dict[str, float]:
    losses = []
    correct = []
    predictions = []
    for start in range(0, targets.numel(), chunk_size):
        batch_inputs = inputs[start : start + chunk_size]
        batch_targets = targets[start : start + chunk_size]
        logits = model(batch_inputs)
        losses.append(torch.nn.functional.cross_entropy(logits, batch_targets, reduction="none"))
        predicted = logits.argmax(dim=1)
        predictions.append(predicted)
        correct.append((predicted == batch_targets).to(logits.dtype))
    loss = torch.cat(losses)
    accuracy = torch.cat(correct)
    predicted = torch.cat(predictions)
    nine = targets == 9
    non_nine_nll = float(loss[~nine].mean())
    nine_nll = float(loss[nine].mean())
    non_nine_accuracy = float(accuracy[~nine].mean())
    nine_accuracy = float(accuracy[nine].mean())
    return {
        "current_nll": (1 - p) * non_nine_nll + p * nine_nll,
        "current_accuracy": (1 - p) * non_nine_accuracy + p * nine_accuracy,
        "p0_nll": non_nine_nll,
        "p0_accuracy": non_nine_accuracy,
        "p1_nll": nine_nll,
        "p1_accuracy": nine_accuracy,
        "worst_panel_nll": max(non_nine_nll, nine_nll),
        **nine_ovr_metrics(predicted, targets, p),
    }


__all__ = [
    "MIXTURE_ASSET_REQUIRED",
    "MIXTURE_PI",
    "ensure_mixture_assets",
    "evaluate_mixture",
    "mixture_asset_unit",
    "mixture_data_config",
    "nine_ovr_metrics",
]
