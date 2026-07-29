"""Validate the score-only reference Fisher and LFU stencil on MNIST."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from src.artifacts import RunStore
from src.config import ExperimentConfig, load_config
from src.initialization import (
    load_replica_bundle_for_config,
    state_dict_hash,
)
from src.mnist_data import (
    ReferenceSamplePlan,
    dataset_targets,
    generate_reference_sample_plan,
    load_mnist_datasets,
)
from src.mnist_model import (
    configure_torch_runtime,
    mnist_losses,
    mnist_nll,
    resolve_device,
    resolve_dtype,
)
from src.parameters import ParameterLayout
from src.reference import (
    ReferenceFisherEstimate,
    ReferenceFisherStore,
    central_fisher_stencil,
    chunked_lfu_estimate,
    chunked_reference_fisher,
    convergence_diagnostics,
    reference_cache_key,
    relative_frobenius_error,
)
from src.seeding import derive_component_seed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--replica-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _normalized(vector: Tensor) -> Tensor:
    norm = torch.linalg.vector_norm(vector)
    if not torch.isfinite(norm) or float(norm) == 0.0:
        raise RuntimeError("calibration produced an invalid zero displacement")
    return vector / norm


def _calibrate_checkpoint(
    initial_model: nn.Module,
    dataset,
    train_targets: Tensor,
    config: ExperimentConfig,
    partitions,
    *,
    p: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, list[Tensor], dict[str, Any]]:
    reference = config.reference
    sample_size = reference.calibration_steps * reference.calibration_batch_size
    seed = derive_component_seed(
        config.replica_seed,
        f"phase3_calibration:p={p:.17g}",
    )
    plan = generate_reference_sample_plan(
        train_targets,
        partitions,
        config.data,
        p=p,
        sample_size=sample_size,
        seed=seed,
        pool="online",
    )
    loader = DataLoader(
        Subset(dataset, plan.observation_indices),
        batch_size=reference.calibration_batch_size,
        shuffle=False,
    )
    calibration_device = (
        torch.device("cpu")
        if config.runtime.deterministic_algorithms
        else device
    )
    model = copy.deepcopy(initial_model).to(
        device=calibration_device,
        dtype=dtype,
    )
    layout = ParameterLayout.from_module(model)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=reference.calibration_learning_rate,
    )
    displacements = []
    losses = []
    model.train()
    for inputs, targets in loader:
        before = layout.flatten_module(model, detach=True)
        inputs = inputs.to(device=calibration_device, dtype=dtype)
        targets = targets.to(device=calibration_device)
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        after = layout.flatten_module(model, detach=True)
        displacements.append(after - before)
        losses.append(float(loss.detach()))

    if len(displacements) != reference.calibration_steps:
        raise RuntimeError("calibration did not execute the configured step count")
    if reference.stencil_direction_count > len(displacements):
        raise RuntimeError("not enough calibration updates for stencil directions")
    directions = [
        _normalized(displacement)
        for displacement in displacements[-reference.stencil_direction_count :]
    ]
    metrics = {
        "seed": seed,
        "device": str(calibration_device),
        "sample_plan_hash": plan.content_hash,
        "steps": len(displacements),
        "batch_size": reference.calibration_batch_size,
        "learning_rate": reference.calibration_learning_rate,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_loss": sum(losses) / len(losses),
        "last_displacement_norms": [
            float(torch.linalg.vector_norm(displacement))
            for displacement in displacements[
                -reference.stencil_direction_count :
            ]
        ],
    }
    return model, directions, metrics


def _cached_reference(
    store: ReferenceFisherStore,
    model: nn.Module,
    dataset,
    plan: ReferenceSamplePlan,
    layout: ParameterLayout,
    config: ExperimentConfig,
    *,
    device: torch.device,
    derivative_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> tuple[ReferenceFisherEstimate, str]:
    key = reference_cache_key(
        model,
        layout,
        plan,
        derivative_dtype=derivative_dtype,
        matrix_dtype=matrix_dtype,
    )
    if store.exists(key):
        return store.load(key, layout), key.digest
    estimate = chunked_reference_fisher(
        model,
        dataset,
        plan,
        mnist_nll,
        layout,
        chunk_size=min(config.reference.chunk_size, plan.sample_size),
        device=device,
        derivative_dtype=derivative_dtype,
        matrix_dtype=matrix_dtype,
        strategy="vmap",
        num_workers=config.initialization.num_workers,
    )
    store.save(key, estimate)
    return estimate, key.digest


def _matrix_metrics(
    lfu,
    stencil: Tensor,
) -> dict[str, float | None]:
    ac = lfu.estimate.amari_chentsov
    residual = lfu.estimate.residual
    full = lfu.estimate.full
    ac_norm = torch.linalg.matrix_norm(ac, ord="fro")
    residual_norm = torch.linalg.matrix_norm(residual, ord="fro")
    denominator = ac_norm * residual_norm
    alignment = (
        None
        if float(denominator) == 0.0
        else float((ac * residual).sum() / denominator)
    )
    return {
        "stencil_fro": float(torch.linalg.matrix_norm(stencil, ord="fro")),
        "ac_fro": float(ac_norm),
        "residual_fro": float(residual_norm),
        "full_fro": float(torch.linalg.matrix_norm(full, ord="fro")),
        "ac_residual_alignment": alignment,
        "ac_relative_error": relative_frobenius_error(ac, stencil),
        "residual_relative_error": relative_frobenius_error(residual, stencil),
        "full_relative_error": relative_frobenius_error(full, stencil),
    }


def _run_p_checkpoint(
    config: ExperimentConfig,
    initial_model: nn.Module,
    partitions,
    train_dataset,
    train_targets: Tensor,
    *,
    p: float,
    store: ReferenceFisherStore,
    device: torch.device,
    training_dtype: torch.dtype,
    derivative_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibrated, update_directions, calibration_metrics = _calibrate_checkpoint(
        initial_model,
        train_dataset,
        train_targets,
        config,
        partitions,
        p=p,
        device=device,
        dtype=training_dtype,
    )
    calibrated = calibrated.to(device=device, dtype=derivative_dtype)
    layout = ParameterLayout.from_module(calibrated)
    directions = [
        _normalized(direction.to(device=device, dtype=derivative_dtype))
        for direction in update_directions
    ]
    reference_seed = derive_component_seed(
        config.replica_seed,
        f"reference_stream:p={p:.17g}",
    )
    full_plan = generate_reference_sample_plan(
        train_targets,
        partitions,
        config.data,
        p=p,
        sample_size=config.reference.sample_size,
        seed=reference_seed,
        pool="reference",
    )

    convergence_matrices = {}
    convergence_entries = []
    for sample_size in config.reference.convergence_sample_sizes:
        plan = full_plan.prefix(sample_size)
        estimate, digest = _cached_reference(
            store,
            calibrated,
            train_dataset,
            plan,
            layout,
            config,
            device=device,
            derivative_dtype=derivative_dtype,
            matrix_dtype=matrix_dtype,
        )
        convergence_matrices[sample_size] = estimate.matrix
        convergence_entries.append(
            {
                "sample_size": sample_size,
                "cache_digest": digest,
                **estimate.diagnostics_mapping(),
            }
        )
    convergence = convergence_diagnostics(convergence_matrices)
    for entry, diagnostic in zip(
        convergence_entries,
        convergence,
        strict=True,
    ):
        entry.update(diagnostic)

    largest_reference = convergence_matrices[config.reference.sample_size]
    _synchronize = (
        (lambda: torch.cuda.synchronize(device))
        if device.type == "cuda"
        else (lambda: None)
    )
    eig_matrix = largest_reference.to(device=device)
    _synchronize()
    eig_start = time.perf_counter()
    eigenvalues = torch.linalg.eigvalsh(eig_matrix)
    _synchronize()
    eig_elapsed = time.perf_counter() - eig_start

    direction_results = []
    for direction_index, direction in enumerate(directions):
        sample_sizes = (
            config.reference.convergence_sample_sizes
            if direction_index == 0
            else [config.reference.sample_size]
        )
        for sample_size in sample_sizes:
            plan = full_plan.prefix(sample_size)
            lfu = chunked_lfu_estimate(
                calibrated,
                train_dataset,
                plan,
                mnist_nll,
                layout,
                direction,
                chunk_size=min(config.reference.chunk_size, sample_size),
                device=device,
                derivative_dtype=derivative_dtype,
                matrix_dtype=matrix_dtype,
                strategy="vmap",
                num_workers=config.initialization.num_workers,
            )
            weighted_results = []
            weighted_derivatives = []
            for epsilon in config.reference.stencil_epsilons:
                stencil = central_fisher_stencil(
                    calibrated,
                    train_dataset,
                    plan,
                    mnist_nll,
                    mnist_losses,
                    layout,
                    direction,
                    epsilon,
                    chunk_size=min(config.reference.chunk_size, sample_size),
                    device=device,
                    derivative_dtype=derivative_dtype,
                    matrix_dtype=matrix_dtype,
                    strategy="vmap",
                    num_workers=config.initialization.num_workers,
                    importance_weighted=True,
                    cache_store=store,
                )
                weighted_derivatives.append(stencil.derivative)
                weighted_results.append(
                    {
                        "epsilon": epsilon,
                        "plus_cache_digest": stencil.plus_cache_digest,
                        "minus_cache_digest": stencil.minus_cache_digest,
                        "plus_weight_mean": stencil.plus.weight_mean,
                        "minus_weight_mean": stencil.minus.weight_mean,
                        "plus_effective_sample_size": (
                            stencil.plus.effective_sample_size
                        ),
                        "minus_effective_sample_size": (
                            stencil.minus.effective_sample_size
                        ),
                        **_matrix_metrics(lfu, stencil.derivative),
                    }
                )
            for index, entry in enumerate(weighted_results):
                entry["relative_to_next_epsilon"] = (
                    None
                    if index + 1 == len(weighted_results)
                    else relative_frobenius_error(
                        weighted_derivatives[index],
                        weighted_derivatives[index + 1],
                    )
                )

            fixed_measure_results = []
            if sample_size == config.reference.sample_size:
                for epsilon in config.reference.stencil_epsilons:
                    stencil = central_fisher_stencil(
                        calibrated,
                        train_dataset,
                        plan,
                        mnist_nll,
                        mnist_losses,
                        layout,
                        direction,
                        epsilon,
                        chunk_size=config.reference.chunk_size,
                        device=device,
                        derivative_dtype=derivative_dtype,
                        matrix_dtype=matrix_dtype,
                        strategy="vmap",
                        num_workers=config.initialization.num_workers,
                        importance_weighted=False,
                        cache_store=store,
                    )
                    fixed_measure_results.append(
                        {
                            "epsilon": epsilon,
                            "plus_cache_digest": stencil.plus_cache_digest,
                            "minus_cache_digest": stencil.minus_cache_digest,
                            **_matrix_metrics(lfu, stencil.derivative),
                        }
                    )

            direction_results.append(
                {
                    "direction_index": direction_index,
                    "sample_size": sample_size,
                    "direction_norm": float(torch.linalg.vector_norm(direction)),
                    "lfu_elapsed_seconds": lfu.elapsed_seconds,
                    "score_gradient_count": lfu.score_gradient_count,
                    "hvp_count": lfu.hvp_count,
                    "weighted_stencils": weighted_results,
                    "fixed_measure_stencils": fixed_measure_results,
                }
            )

    checkpoint_artifact = {
        "p": p,
        "model_state": {
            name: tensor.detach().cpu()
            for name, tensor in calibrated.state_dict().items()
        },
        "parameter_layout": layout.metadata(),
        "directions": [direction.detach().cpu() for direction in directions],
        "reference_plan": full_plan.to_mapping(),
    }
    result = {
        "p": p,
        "checkpoint_hash": state_dict_hash(calibrated.state_dict()),
        "calibration": calibration_metrics,
        "reference_seed": reference_seed,
        "reference_plan_hash": full_plan.content_hash,
        "convergence": convergence_entries,
        "eigendecomposition": {
            "elapsed_seconds": eig_elapsed,
            "minimum_eigenvalue": float(eigenvalues.min()),
            "maximum_eigenvalue": float(eigenvalues.max()),
        },
        "directions": direction_results,
    }
    return result, checkpoint_artifact


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=(
            config.runtime.deterministic_algorithms
            and config.runtime.device in {"cuda", "auto"}
        ),
    )
    cache_parent = Path(config.cache_root).parent
    data_root = arguments.data_root or cache_parent / "datasets"
    replica_root = arguments.replica_root or cache_parent / "replicas"
    reference_root = arguments.reference_root or cache_parent / "references"
    output_root = arguments.output_root or cache_parent / "phase3_runs"
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.training_dtype)
    derivative_dtype = resolve_dtype(config.reference.derivative_dtype)
    matrix_dtype = resolve_dtype(config.runtime.matrix_dtype)

    session = RunStore(output_root).begin(
        config,
        Path(__file__).parents[1],
        resume=arguments.resume,
    )
    train_dataset, _ = load_mnist_datasets(data_root, download=False)
    train_targets = dataset_targets(train_dataset)
    loaded = load_replica_bundle_for_config(
        replica_root,
        config,
        device=device,
    )
    store = ReferenceFisherStore(reference_root)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    results = []
    checkpoint_artifacts = {}
    for p in config.reference.stencil_p_values:
        result, artifact = _run_p_checkpoint(
            config,
            loaded.model,
            loaded.partitions,
            train_dataset,
            train_targets,
            p=p,
            store=store,
            device=device,
            training_dtype=training_dtype,
            derivative_dtype=derivative_dtype,
            matrix_dtype=matrix_dtype,
        )
        results.append(result)
        checkpoint_artifacts[f"{p:.17g}"] = artifact

    metrics = {
        "reference_fisher_schema_version": 1,
        "replica_bundle_id": loaded.metadata["bundle_id"],
        "device": str(device),
        "training_dtype": str(training_dtype),
        "derivative_dtype": str(derivative_dtype),
        "matrix_dtype": str(matrix_dtype),
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "checkpoints": results,
    }
    session.write_json("phase3_metrics.json", metrics)
    session.write_torch("phase3_checkpoints.pt", checkpoint_artifacts)
    destination = session.complete(
        ["phase3_metrics.json", "phase3_checkpoints.pt"]
    )
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "path": str(destination),
                "checkpoints": len(results),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
