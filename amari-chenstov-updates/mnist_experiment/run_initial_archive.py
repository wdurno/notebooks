"""Build an immutable, oracle-free rank-8 Fisher archive at p=0."""

from __future__ import annotations

import argparse
import copy
import json
import resource
import time
from pathlib import Path

import torch

from src.artifacts import RunStore
from src.config import ExperimentConfig, load_config
from src.hybrid import (
    INITIAL_ARCHIVE_SOURCE_SCHEMA_VERSION,
    HybridArchiveState,
)
from src.initialization import load_replica_bundle_for_config
from src.lanczos_wrapper import approximate_low_rank_diagonal
from src.mnist_data import (
    dataset_targets,
    generate_reference_sample_plan,
    load_mnist_datasets,
)
from src.mnist_model import (
    configure_torch_runtime,
    mnist_nll,
    resolve_device,
    resolve_dtype,
)
from src.parameters import ParameterLayout
from src.reference import adaptive_reference_fisher
from src.seeding import derive_component_seed


INITIAL_ARCHIVE_METRIC_SCHEMA_VERSION = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--replica-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_config(config: ExperimentConfig) -> None:
    if config.estimator.representation != "low_rank_diagonal":
        raise ValueError("initial archive requires low-rank-plus-diagonal Fisher")
    if config.estimator.low_rank is None or config.estimator.low_rank < 1:
        raise ValueError("initial archive requires a positive Fisher rank")
    if config.controller.oracle_mode != "none":
        raise ValueError("initial archive must be oracle-free")
    if config.controller.reference_optimum_artifact is not None:
        raise ValueError("initial archive cannot reference an optimum path")


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    _validate_config(config)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=(
            config.runtime.deterministic_algorithms
            and config.runtime.device in {"cuda", "auto"}
        ),
    )
    repo_root = Path(__file__).parents[1]
    cache_parent = Path(config.cache_root).parent
    data_root = arguments.data_root or cache_parent / "datasets"
    replica_root = arguments.replica_root or cache_parent / "replicas"
    output_root = arguments.output_root or Path(config.cache_root)
    device = resolve_device(config.runtime.device)
    derivative_dtype = resolve_dtype(config.reference.derivative_dtype)
    matrix_dtype = resolve_dtype(config.runtime.matrix_dtype)
    session = RunStore(output_root).begin(
        config,
        repo_root,
        resume=arguments.resume,
    )

    started = time.perf_counter()
    train_dataset, _ = load_mnist_datasets(data_root, download=False)
    train_targets = dataset_targets(train_dataset)
    loaded = load_replica_bundle_for_config(replica_root, config, device=device)
    anchor = loaded.layout.flatten_module(loaded.model, detach=True).cpu()
    model = copy.deepcopy(loaded.model).to(device=device, dtype=derivative_dtype)
    layout = ParameterLayout.from_module(model)

    plan_seed = derive_component_seed(
        config.replica_seed,
        "phase4_reference:step=0:p=0",
    )
    plan = generate_reference_sample_plan(
        train_targets,
        loaded.partitions,
        config.data,
        p=0.0,
        sample_size=config.reference.sample_size,
        seed=plan_seed,
        pool="reference",
    )
    print("Phase 7 p=0 Fisher archive started", flush=True)
    estimate = adaptive_reference_fisher(
        model,
        train_dataset,
        plan,
        mnist_nll,
        layout,
        chunk_size=config.reference.chunk_size,
        minimum_chunks=config.reference.convergence_min_chunks,
        sigma=config.reference.convergence_sigma,
        relative_epsilon=config.reference.convergence_relative_epsilon,
        absolute_epsilon=config.reference.convergence_absolute_epsilon,
        device=device,
        derivative_dtype=derivative_dtype,
        matrix_dtype=matrix_dtype,
        strategy="vmap",
        num_workers=config.initialization.num_workers,
    )
    matrix = estimate.matrix.to(dtype=matrix_dtype)
    matrix = (matrix + matrix.mT) / 2
    projection_seed = derive_component_seed(
        config.replica_seed,
        "plan3_phase7_initial_lanczos",
    )
    approximation = approximate_low_rank_diagonal(
        lambda vectors: matrix @ vectors,
        torch.diagonal(matrix),
        rank=config.estimator.low_rank,
        seed=projection_seed,
    )
    state = HybridArchiveState(
        anchor=anchor,
        fisher=approximation.representation,
        initial_anchor_observations=config.data.initialization_size,
        initial_fisher_score_observations=estimate.sample_count,
    )
    state.validate()
    elapsed = time.perf_counter() - started
    artifact = {
        "schema_version": INITIAL_ARCHIVE_SOURCE_SCHEMA_VERSION,
        "kind": "plan3_initial_archive",
        "replica_id": config.replica_id,
        "replica_bundle_id": loaded.metadata["bundle_id"],
        "stream_plan_hash": loaded.stream_plan.content_hash,
        "reference_plan": plan.prefix(estimate.sample_count).to_mapping(),
        "archive_state": state.to_mapping(),
        "fisher_diagnostics": estimate.diagnostics_mapping(),
        "lanczos_diagnostics": approximation.diagnostics.mapping(),
    }
    metrics = {
        "plan3_initial_archive_metric_schema_version": (
            INITIAL_ARCHIVE_METRIC_SCHEMA_VERSION
        ),
        "replica_id": config.replica_id,
        "replica_bundle_id": loaded.metadata["bundle_id"],
        "stream_plan_hash": loaded.stream_plan.content_hash,
        "oracle_free": True,
        "p": 0.0,
        "reference_plan_hash": plan.prefix(estimate.sample_count).content_hash,
        "configured_score_budget": config.reference.sample_size,
        "score_gradient_count": estimate.score_gradient_count,
        "fisher": estimate.diagnostics_mapping(),
        "lanczos": approximation.diagnostics.mapping(),
        "parameter_count": layout.total_numel,
        "represented_rank": state.rank,
        "represented_trace": float(
            state.fisher.factor.square().sum()
            + state.fisher.residual_diagonal.sum()
        ),
        "wall_time_seconds": elapsed,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    session.write_torch("initial_archive.pt", artifact)
    session.write_json("initial_archive_metrics.json", metrics)
    destination = session.complete(
        ("initial_archive.pt", "initial_archive_metrics.json")
    )
    print("Phase 7 p=0 Fisher archive completed", flush=True)
    print(
        json.dumps(
            {
                "path": str(destination),
                "score_gradient_count": estimate.score_gradient_count,
                "represented_rank": state.rank,
                "wall_time_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
