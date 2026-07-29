"""Build an immutable paired MNIST initialization and stream bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.initialization import (
    fit_p0_initialization,
    replica_bundle_id,
    save_replica_bundle,
)
from src.mnist_data import (
    dataset_targets,
    generate_mixture_stream,
    load_mnist_datasets,
    partition_mnist,
)
from src.mnist_model import (
    build_canonical_model,
    configure_torch_runtime,
    resolve_device,
    resolve_dtype,
)
from src.seeding import derive_seed_map


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=False,
    )
    cache_parent = Path(config.cache_root).parent
    data_root = arguments.data_root or cache_parent / "datasets"
    output_root = arguments.output_root or cache_parent / "replicas"
    device = resolve_device(config.runtime.device)
    dtype = resolve_dtype(config.runtime.training_dtype)
    seeds = derive_seed_map(config.replica_seed)

    train_dataset, test_dataset = load_mnist_datasets(
        data_root,
        download=arguments.download,
    )
    train_targets = dataset_targets(train_dataset)
    test_targets = dataset_targets(test_dataset)
    partitions = partition_mnist(
        train_targets,
        test_targets,
        config.data,
        replica_seed=config.replica_seed,
    )
    stream_plan = generate_mixture_stream(
        train_targets,
        partitions,
        config.data,
        seed=seeds["online_stream"],
    )
    model, layout = build_canonical_model(
        seeds["initialization"],
        device=device,
        dtype=dtype,
    )
    initialization = fit_p0_initialization(
        model,
        train_dataset,
        test_dataset,
        partitions,
        config.initialization,
        loader_seed=seeds["initialization_loader"],
        device=device,
        dtype=dtype,
    )
    destination = save_replica_bundle(
        output_root,
        config,
        model,
        layout,
        partitions,
        stream_plan,
        initialization,
        device=device,
        repo_root=Path(__file__).parents[1],
    )
    print(
        json.dumps(
            {
                "bundle_id": replica_bundle_id(config),
                "path": str(destination),
                "metrics": initialization.final_metrics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
