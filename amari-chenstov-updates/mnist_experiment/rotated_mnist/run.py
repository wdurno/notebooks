"""Execute one immutable Plan 5 rotated-MNIST trajectory."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import random
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from src.config import OptimizerConfig
from src.ewc import build_optimizer, take_ewc_proposal
from src.initialization import evaluate_classifier, state_dict_hash
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import (
    build_canonical_model,
    configure_torch_runtime,
    resolve_device,
    resolve_dtype,
)
from src.parameters import ParameterLayout
from src.representations import DiagonalFisher
from src.seeding import derive_component_seed

from .artifacts import RotatedRunStore
from .config import RotatedExperimentConfig, load_config
from .data import (
    RotatedDatasetView,
    generate_rotated_stream,
    partition_all_digit_mnist,
)
from .schedule import resolve_rotation_schedule
from .transform import tensor_content_hash


CALIBRATION_BINS = 10


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("cache/mnist_experiment/datasets"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/mnist_experiment/rotated_mnist/runs"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = (base_seed + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=partial(_seed_worker, base_seed=seed),
        persistent_workers=num_workers > 0,
    )


def _state_dict_cpu(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _prefix(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _fit_upright_initializer(
    model: nn.Module,
    train_dataset: Dataset,
    evaluation_dataset: Dataset,
    initialization_indices: tuple[int, ...],
    evaluation_indices: tuple[int, ...],
    config: RotatedExperimentConfig,
    *,
    nine_prevalence: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    initialization_targets = dataset_targets(train_dataset)[
        list(initialization_indices)
    ]
    if set(initialization_targets.tolist()) != set(range(10)):
        raise ValueError("rotated initializer requires all ten MNIST classes")
    training = Subset(train_dataset, initialization_indices)
    evaluation = RotatedDatasetView(
        evaluation_dataset,
        evaluation_indices,
        angle_degrees=0.0,
        rotation_config=config.rotation,
    )
    loader_seed = derive_component_seed(
        config.replica_seed, "plan5_initialization_loader"
    )
    loader = _loader(
        training,
        batch_size=config.initialization.batch_size,
        shuffle=True,
        seed=loader_seed,
        num_workers=config.runtime.num_workers,
    )
    arguments = {
        "lr": config.initialization.learning_rate,
        "weight_decay": config.initialization.weight_decay,
    }
    optimizer: torch.optim.Optimizer
    if config.initialization.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **arguments)
    else:
        optimizer = torch.optim.SGD(model.parameters(), **arguments)

    started = time.perf_counter()
    history = []
    stopped_on_target = False
    for epoch in range(1, config.initialization.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=dtype)
            targets = targets.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * targets.numel()
            count += targets.numel()
        metrics = evaluate_classifier(
            model,
            evaluation,
            batch_size=config.initialization.batch_size,
            device=device,
            dtype=dtype,
            num_workers=config.runtime.num_workers,
            calibration_bins=CALIBRATION_BINS,
            nine_prevalence=nine_prevalence,
        )
        history.append(
            {"epoch": epoch, "training_nll": loss_sum / count, **metrics}
        )
        target = config.initialization.target_accuracy
        if target is not None and float(metrics["accuracy"]) >= target:
            stopped_on_target = True
            break
    return {
        "epochs_completed": len(history),
        "stopped_on_target": stopped_on_target,
        "wall_time_seconds": time.perf_counter() - started,
        "history": history,
        "final_metrics": history[-1],
    }


def _learner_optimizer_config(config: RotatedExperimentConfig) -> OptimizerConfig:
    learner = config.learner
    result = OptimizerConfig(
        name=learner.optimizer,
        learning_rate=learner.learning_rate,
        inner_steps=learner.inner_steps,
        ewc_strength=1.0,
        lbfgs_history_size=learner.lbfgs_history_size,
        lbfgs_max_eval_factor=learner.lbfgs_max_eval_factor,
        lbfgs_tolerance_grad=learner.lbfgs_tolerance_grad,
        lbfgs_tolerance_change=learner.lbfgs_tolerance_change,
        lbfgs_line_search_fn=learner.lbfgs_line_search_fn,
    )
    result.validate()
    return result


def _evaluate_angles(
    model: nn.Module,
    evaluation_dataset: Dataset,
    evaluation_indices: tuple[int, ...],
    config: RotatedExperimentConfig,
    *,
    current_angle: float,
    nine_prevalence: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    angles = {
        "current": current_angle,
        "panel_000": 0.0,
        "panel_015": 15.0,
        "panel_030": 30.0,
    }
    output: dict[str, Any] = {}
    for name, angle in angles.items():
        view = RotatedDatasetView(
            evaluation_dataset,
            evaluation_indices,
            angle_degrees=angle,
            rotation_config=config.rotation,
        )
        metrics = evaluate_classifier(
            model,
            view,
            batch_size=config.initialization.batch_size,
            device=device,
            dtype=dtype,
            num_workers=config.runtime.num_workers,
            calibration_bins=CALIBRATION_BINS,
            nine_prevalence=nine_prevalence,
        )
        output.update(_prefix(name, metrics))
    return output


def _trajectory_hash(parameters: Tensor, displacements: Tensor, stream_hash: str) -> str:
    payload = (
        tensor_content_hash(parameters)
        + tensor_content_hash(displacements)
        + stream_hash
    )
    import hashlib

    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def run_experiment(
    config: RotatedExperimentConfig,
    *,
    data_root: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    download: bool = False,
    resume: bool = False,
) -> Path:
    config.validate()
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=False,
    )
    device = resolve_device(config.runtime.device)
    dtype = resolve_dtype(config.runtime.dtype)
    store = RotatedRunStore(output_root)
    session = store.begin(config, repo_root, resume=resume)
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=download)
    train_targets = dataset_targets(train_dataset)
    test_targets = dataset_targets(test_dataset)
    nine_prevalence = float((train_targets == 9).double().mean())
    partitions = partition_all_digit_mnist(
        train_targets,
        test_targets,
        config.data,
        replica_seed=config.replica_seed,
    )
    schedule = resolve_rotation_schedule(config.rotation)
    stream_plan, stream_inputs, stream_targets = generate_rotated_stream(
        train_dataset,
        train_targets,
        partitions,
        schedule,
        config.data,
        config.rotation,
        replica_seed=config.replica_seed,
    )
    session.write_json("partitions.json", partitions.to_mapping())
    session.write_json("stream_plan.json", stream_plan.to_mapping())
    session.write_torch(
        "stream_tensors.pt", {"inputs": stream_inputs, "targets": stream_targets}
    )

    model_seed = derive_component_seed(
        config.replica_seed, "plan5_model_initialization"
    )
    model, layout = build_canonical_model(model_seed, device=device, dtype=dtype)
    initialization = _fit_upright_initializer(
        model,
        train_dataset,
        test_dataset,
        partitions.initialization,
        partitions.evaluation,
        config,
        nine_prevalence=nine_prevalence,
        device=device,
        dtype=dtype,
    )
    initial_state = _state_dict_cpu(model)
    initial_state_hash = state_dict_hash(initial_state)
    optimizer_config = _learner_optimizer_config(config)
    optimizer = build_optimizer(model, optimizer_config)
    zero_fisher = DiagonalFisher(
        torch.zeros(layout.total_numel, device=device, dtype=dtype)
    )

    parameters = []
    displacements = []
    rows = []
    cumulative_nines = 0
    started = time.perf_counter()
    progress = tqdm(
        range(schedule.num_points),
        desc="rotated current-only smoke",
        unit="step",
        leave=False,
    )
    for step in progress:
        angle = schedule.angles_degrees[step]
        parameter_before = layout.flatten_module(model, detach=True)
        parameters.append(parameter_before.cpu())
        evaluations = _evaluate_angles(
            model,
            test_dataset,
            partitions.evaluation,
            config,
            current_angle=angle,
            nine_prevalence=nine_prevalence,
            device=device,
            dtype=dtype,
        )
        proposal_mapping = None
        if step < schedule.num_transitions:
            inputs = stream_inputs[step].to(device=device, dtype=dtype)
            targets = stream_targets[step].to(device=device)
            proposal = take_ewc_proposal(
                model,
                layout,
                inputs,
                targets,
                zero_fisher,
                optimizer_config,
                optimizer,
                adaptation_weight=1.0,
            )
            displacement = (
                layout.flatten_module(model, detach=True) - parameter_before
            ).cpu()
            if not torch.equal(displacement, proposal.displacement):
                raise RuntimeError("current-only proposal displacement was altered")
            displacements.append(displacement)
            proposal_mapping = proposal.metrics_mapping()
        labels = stream_plan.class_labels[step]
        rows.append(
            {
                "step": step,
                "angle_degrees": angle,
                "leg_id": schedule.leg_ids[step],
                "direction_to_next": schedule.directions_to_next[step],
                "knot": schedule.knot_flags[step],
                "cumulative_angular_degrees": schedule.cumulative_degrees[step],
                "observations_before_evaluation": step
                * config.data.samples_per_step,
                "observed_nines_before_evaluation": cumulative_nines,
                "condition": config.learner.condition,
                "parameter_hash": tensor_content_hash(parameter_before.cpu()),
                "proposal": proposal_mapping,
                **evaluations,
            }
        )
        if step < schedule.num_transitions:
            cumulative_nines += sum(label == 9 for label in labels)

    parameter_tensor = torch.stack(parameters)
    displacement_tensor = torch.stack(displacements)
    if not torch.equal(
        displacement_tensor,
        parameter_tensor[1:] - parameter_tensor[:-1],
    ):
        raise RuntimeError("rotated trajectory displacement identity failed")
    final_state = _state_dict_cpu(model)
    final_state_hash = state_dict_hash(final_state)
    trajectory_hash = _trajectory_hash(
        parameter_tensor, displacement_tensor, stream_plan.content_hash
    )
    elapsed = time.perf_counter() - started

    session.write_json("initialization_metrics.json", initialization)
    session.write_json("trajectory_metrics.json", rows)
    session.write_torch(
        "trajectory.pt",
        {"parameters": parameter_tensor, "displacements": displacement_tensor},
    )
    session.write_torch(
        "model_states.pt", {"initial": initial_state, "final": final_state}
    )
    session.write_json(
        "run_summary.json",
        {
            "artifact_schema_version": config.artifact_schema_version,
            "metric_schema_version": config.metric_schema_version,
            "condition": config.learner.condition,
            "schedule_hash": schedule.content_hash,
            "stream_plan_hash": stream_plan.content_hash,
            "partition_hash": partitions.content_hash,
            "trajectory_hash": trajectory_hash,
            "initial_model_state_hash": initial_state_hash,
            "final_model_state_hash": final_state_hash,
            "parameter_count": layout.total_numel,
            "num_points": schedule.num_points,
            "num_transitions": schedule.num_transitions,
            "samples_per_step": config.data.samples_per_step,
            "master_samples_per_step": config.data.stream_width,
            "nine_prevalence": nine_prevalence,
            "trajectory_wall_time_seconds": elapsed,
            "final_current_nine_ovr_accuracy": rows[-1][
                "current_nine_ovr_accuracy"
            ],
            "final_current_environment_accuracy": rows[-1][
                "current_environment_accuracy"
            ],
        },
    )
    return session.complete()


def main() -> None:
    arguments = parse_arguments()
    config = load_config(arguments.config)
    repo_root = Path(__file__).parents[2]
    path = run_experiment(
        config,
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        repo_root=repo_root,
        download=arguments.download,
        resume=arguments.resume,
    )
    print(
        json.dumps(
            {"run_id": config.run_id, "path": str(path), "status": "completed"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
