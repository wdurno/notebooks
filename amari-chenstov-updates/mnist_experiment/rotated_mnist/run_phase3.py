"""Execute the paired low-data transfer pilot from Plan 5 Phase 3."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import resource
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.derivatives import per_sample_derivatives
from src.ewc import build_optimizer, take_ewc_proposal
from src.fisher import empirical_fisher
from src.hybrid import blend_archive_fisher
from src.initialization import evaluate_classifier, state_dict_hash
from src.lanczos_wrapper import approximate_low_rank_diagonal
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import (
    build_canonical_model,
    configure_torch_runtime,
    mnist_nll,
    resolve_device,
    resolve_dtype,
)
from src.parameters import ParameterLayout
from src.representations import DiagonalFisher, LowRankDiagonalFisher
from src.seeding import derive_component_seed

from .data import (
    RotatedDatasetView,
    generate_rotated_stream,
    partition_all_digit_mnist,
)
from .phase3_artifacts import PHASE3_REQUIRED_ARTIFACTS, RotatedPhase3RunStore
from .phase3_config import (
    PHASE3_CONDITIONS,
    RotatedPhase3Config,
    load_phase3_config,
)
from .run import (
    CALIBRATION_BINS,
    _fit_upright_initializer,
    _learner_optimizer_config,
    _state_dict_cpu,
)
from .schedule import resolve_rotation_schedule
from .transform import tensor_content_hash


PERSISTENT_ENVIRONMENT_THRESHOLDS = (0.6, 0.7, 0.8)


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
        default=Path("cache/mnist_experiment/rotated_mnist/phase3"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _confusion_matrix(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    num_workers: int,
) -> Tensor:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    was_training = model.training
    model.eval()
    confusion = torch.zeros(10, 10, dtype=torch.long)
    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs.to(device=device, dtype=dtype)).argmax(dim=1)
            encoded = targets.to(device=device) * 10 + predictions
            confusion.add_(
                torch.bincount(encoded, minlength=100).reshape(10, 10).cpu()
            )
    model.train(was_training)
    return confusion


def _classwise_metrics(confusion: Tensor) -> dict[str, Any]:
    if confusion.shape != (10, 10) or (confusion < 0).any():
        raise ValueError("confusion matrix must be nonnegative with shape (10, 10)")
    true_positive = torch.diagonal(confusion).to(torch.float64)
    actual = confusion.sum(dim=1).to(torch.float64)
    predicted = confusion.sum(dim=0).to(torch.float64)
    recall = true_positive / actual
    precision_values = []
    for value, denominator in zip(true_positive, predicted, strict=True):
        precision_values.append(
            None if float(denominator) == 0.0 else float(value / denominator)
        )
    return {
        "confusion_matrix": confusion.tolist(),
        "per_class_precision": precision_values,
        "per_class_recall": [float(value) for value in recall],
        "worst_class_recall": float(recall.min()),
        "worst_class_label": int(torch.argmin(recall)),
    }


def _evaluate_panel(
    model: nn.Module,
    evaluation_dataset: Dataset,
    evaluation_indices: tuple[int, ...],
    config: RotatedPhase3Config,
    *,
    current_angle: float,
    nine_prevalence: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    requested = {
        "current": current_angle,
        "panel_000": 0.0,
        "panel_015": 15.0,
        "panel_030": 30.0,
    }
    by_angle: dict[float, dict[str, Any]] = {}
    for angle in dict.fromkeys(requested.values()):
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
        # The fixed all-digit environment uses ordinary multiclass accuracy.
        metrics["nine_prevalence_adjusted_environment_accuracy"] = metrics.pop(
            "environment_accuracy"
        )
        metrics["environment_accuracy"] = metrics["accuracy"]
        confusion = _confusion_matrix(
            model,
            view,
            batch_size=config.initialization.batch_size,
            device=device,
            dtype=dtype,
            num_workers=config.runtime.num_workers,
        )
        by_angle[angle] = {**metrics, **_classwise_metrics(confusion)}
    output: dict[str, Any] = {}
    for name, angle in requested.items():
        output.update(
            {f"{name}_{metric}": value for metric, value in by_angle[angle].items()}
        )
    return output


def _estimate_initial_fisher(
    model: nn.Module,
    layout: ParameterLayout,
    train_dataset: Dataset,
    indices: tuple[int, ...],
    config: RotatedPhase3Config,
    *,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> tuple[Tensor, dict[str, Any]]:
    selected = tuple(indices[: config.fisher.initial_sample_size])
    view = RotatedDatasetView(
        train_dataset,
        selected,
        angle_degrees=0.0,
        rotation_config=config.rotation,
    )
    loader = DataLoader(
        view,
        batch_size=config.fisher.chunk_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        persistent_workers=config.runtime.num_workers > 0,
    )
    accumulator = torch.zeros(
        layout.total_numel,
        layout.total_numel,
        device=device,
        dtype=matrix_dtype,
    )
    score_count = 0
    _synchronize(device)
    started = time.perf_counter()
    progress = tqdm(loader, desc="Phase 3 initial Fisher", unit="chunk", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device=device, dtype=training_dtype)
        targets = targets.to(device=device)
        gradients = per_sample_derivatives(
            model,
            inputs,
            targets,
            mnist_nll,
            layout,
            strategy="vmap",
        ).gradients.to(dtype=matrix_dtype)
        accumulator.add_(gradients.mT @ gradients)
        score_count += gradients.shape[0]
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if score_count != len(selected):
        raise RuntimeError("initial Fisher score count differs from its sample plan")
    fisher = accumulator / score_count
    fisher = (fisher + fisher.mT) / 2
    if not torch.isfinite(fisher).all():
        raise RuntimeError("initial Fisher estimate is nonfinite")
    return fisher, {
        "sample_count": score_count,
        "chunk_size": config.fisher.chunk_size,
        "wall_time_seconds": elapsed,
        "trace": float(torch.trace(fisher)),
        "frobenius_norm": float(torch.linalg.matrix_norm(fisher, ord="fro")),
        "sample_indices_hash": tensor_content_hash(
            torch.as_tensor(selected, dtype=torch.long)
        ),
    }


def _fresh_fisher(
    model: nn.Module,
    layout: ParameterLayout,
    inputs: Tensor,
    targets: Tensor,
    *,
    matrix_dtype: torch.dtype,
) -> Tensor:
    gradients = per_sample_derivatives(
        model,
        inputs,
        targets,
        mnist_nll,
        layout,
        strategy="vmap",
    ).gradients.to(dtype=matrix_dtype)
    return empirical_fisher(gradients)


def _normalized_auc(rows: list[dict[str, Any]], field: str) -> float:
    x = torch.tensor(
        [row["observations_before_evaluation"] for row in rows],
        dtype=torch.float64,
    )
    y = torch.tensor([row[field] for row in rows], dtype=torch.float64)
    width = float(x[-1] - x[0])
    if width <= 0.0:
        raise ValueError("trajectory AUC requires increasing observation exposure")
    return float(torch.trapezoid(y, x=x) / width)


def _persistent_thresholds(rows: list[dict[str, Any]]) -> dict[str, int | None]:
    values = [float(row["current_environment_accuracy"]) for row in rows]
    observations = [int(row["observations_before_evaluation"]) for row in rows]
    output: dict[str, int | None] = {}
    for threshold in PERSISTENT_ENVIRONMENT_THRESHOLDS:
        crossing = next(
            (
                observations[index]
                for index in range(len(values))
                if all(value >= threshold for value in values[index:])
            ),
            None,
        )
        output[f"persistent_env_accuracy_{threshold:.1f}"] = crossing
    return output


def _condition_trajectory(
    condition: str,
    initial_model: nn.Module,
    initial_fisher: LowRankDiagonalFisher,
    train_dataset: Dataset,
    test_dataset: Dataset,
    evaluation_indices: tuple[int, ...],
    stream_inputs: Tensor,
    stream_targets: Tensor,
    stream_plan,
    config: RotatedPhase3Config,
    *,
    nine_prevalence: float,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], dict[str, Tensor], dict[str, Any], dict[str, Tensor]]:
    if condition not in PHASE3_CONDITIONS:
        raise ValueError(f"unsupported Phase 3 condition: {condition}")
    model = copy.deepcopy(initial_model).to(device=device, dtype=training_dtype)
    layout = ParameterLayout.from_module(model)
    optimizer_config = _learner_optimizer_config(config)
    optimizer = build_optimizer(model, optimizer_config)
    fisher = initial_fisher.to(device=device, dtype=matrix_dtype)
    zero_fisher = DiagonalFisher(
        torch.zeros(layout.total_numel, device=device, dtype=training_dtype)
    )
    initial_state = _state_dict_cpu(model)
    parameters: list[Tensor] = []
    displacements: list[Tensor] = []
    rows: list[dict[str, Any]] = []
    score_gradient_count = 0
    optimizer_iterations = 0
    optimizer_evaluations = 0
    evaluation_wall_time = 0.0
    fisher_update_wall_time = 0.0
    learner_optimization_wall_time = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    progress = tqdm(
        range(stream_plan.schedule.num_points),
        desc=f"Phase 3 {condition}",
        unit="step",
        leave=False,
    )
    for step in progress:
        angle = stream_plan.schedule.angles_degrees[step]
        parameter_before = layout.flatten_module(model, detach=True)
        parameters.append(parameter_before.cpu())
        _synchronize(device)
        evaluation_started = time.perf_counter()
        evaluations = _evaluate_panel(
            model,
            test_dataset,
            evaluation_indices,
            config,
            current_angle=angle,
            nine_prevalence=nine_prevalence,
            device=device,
            dtype=training_dtype,
        )
        _synchronize(device)
        evaluation_wall_time += time.perf_counter() - evaluation_started
        proposal_mapping = None
        fisher_mapping = None
        if step < stream_plan.schedule.num_transitions:
            inputs = stream_inputs[step].to(device=device, dtype=training_dtype)
            targets = stream_targets[step].to(device=device)
            if condition == "ewc_fixed_pi005":
                _synchronize(device)
                fisher_started = time.perf_counter()
                fresh = _fresh_fisher(
                    model,
                    layout,
                    inputs,
                    targets,
                    matrix_dtype=matrix_dtype,
                )
                score_gradient_count += targets.numel()
                if step == 0:
                    fisher_mapping = {
                        "update": "initial_summary_only",
                        "blend_gain": config.fisher.fixed_pi,
                        "previous_trace": float(fisher.diagonal_vector().sum()),
                        "fresh_trace": float(torch.trace(fresh)),
                        "candidate_trace": float(fisher.diagonal_vector().sum()),
                        "lanczos": None,
                    }
                else:
                    update = blend_archive_fisher(
                        fisher,
                        fresh,
                        blend_gain=config.fisher.fixed_pi,
                        rank=config.fisher.rank,
                        lanczos_seed=derive_component_seed(
                            config.replica_seed,
                            f"plan5_phase3_online_lanczos:step={step}",
                        ),
                    )
                    fisher = update.representation
                    fisher_mapping = {
                        "update": "direct_ema",
                        "blend_gain": update.blend_gain,
                        "previous_trace": update.previous_trace,
                        "fresh_trace": update.fresh_trace,
                        "candidate_trace": update.candidate_trace,
                        "lanczos": update.lanczos.mapping(),
                    }
                _synchronize(device)
                fisher_update_wall_time += time.perf_counter() - fisher_started
                proposal_fisher = fisher.to(dtype=training_dtype)
                adaptation_weight = config.fisher.fixed_pi
            else:
                proposal_fisher = zero_fisher
                adaptation_weight = 1.0
            _synchronize(device)
            optimization_started = time.perf_counter()
            proposal = take_ewc_proposal(
                model,
                layout,
                inputs,
                targets,
                proposal_fisher,
                optimizer_config,
                optimizer,
                adaptation_weight=adaptation_weight,
                penalty_anchor=parameter_before,
            )
            _synchronize(device)
            learner_optimization_wall_time += (
                time.perf_counter() - optimization_started
            )
            displacement = (
                layout.flatten_module(model, detach=True) - parameter_before
            ).cpu()
            if not torch.equal(displacement, proposal.displacement):
                raise RuntimeError("Phase 3 proposal displacement was altered")
            displacements.append(displacement)
            proposal_mapping = proposal.metrics_mapping()
            optimizer_iterations += proposal.optimizer_iterations
            optimizer_evaluations += proposal.optimizer_function_evaluations
        rows.append(
            {
                "step": step,
                "condition": condition,
                "angle_degrees": angle,
                "leg_id": stream_plan.schedule.leg_ids[step],
                "direction_to_next": stream_plan.schedule.directions_to_next[step],
                "knot": stream_plan.schedule.knot_flags[step],
                "cumulative_angular_degrees": (
                    stream_plan.schedule.cumulative_degrees[step]
                ),
                "observations_before_evaluation": step
                * config.data.samples_per_step,
                "parameter_hash": tensor_content_hash(parameter_before.cpu()),
                "proposal": proposal_mapping,
                "fisher_update": fisher_mapping,
                **evaluations,
            }
        )
    _synchronize(device)
    elapsed = time.perf_counter() - started
    parameter_tensor = torch.stack(parameters)
    displacement_tensor = torch.stack(displacements)
    if not torch.equal(
        displacement_tensor, parameter_tensor[1:] - parameter_tensor[:-1]
    ):
        raise RuntimeError("Phase 3 trajectory displacement identity failed")
    final_state = _state_dict_cpu(model)
    summary = {
        "condition": condition,
        "trajectory_wall_time_seconds": elapsed,
        "optimizer_iterations": optimizer_iterations,
        "optimizer_function_evaluations": optimizer_evaluations,
        "score_gradient_count": score_gradient_count,
        "evaluation_wall_time_seconds": evaluation_wall_time,
        "fisher_update_wall_time_seconds": fisher_update_wall_time,
        "learner_optimization_wall_time_seconds": (
            learner_optimization_wall_time
        ),
        "environment_accuracy_auc": _normalized_auc(
            rows, "current_environment_accuracy"
        ),
        "environment_nll_auc": _normalized_auc(rows, "current_nll"),
        "final_current_environment_accuracy": rows[-1][
            "current_environment_accuracy"
        ],
        "final_current_nll": rows[-1]["current_nll"],
        "final_upright_environment_accuracy": rows[-1][
            "panel_000_environment_accuracy"
        ],
        "final_worst_class_recall": rows[-1]["current_worst_class_recall"],
        "fisher_summary_bytes": (
            fisher.storage_bytes() if condition == "ewc_fixed_pi005" else 0
        ),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "peak_process_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        * 1024,
        **_persistent_thresholds(rows),
    }
    return (
        rows,
        {"parameters": parameter_tensor, "displacements": displacement_tensor},
        summary,
        {"initial": initial_state, "final": final_state},
    )


def run_phase3(
    config: RotatedPhase3Config,
    *,
    data_root: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    download: bool = False,
    resume: bool = False,
) -> Path:
    config.validate()
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.dtype)
    matrix_dtype = resolve_dtype(config.fisher.matrix_dtype)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=device.type == "cuda",
    )
    session = RotatedPhase3RunStore(output_root).begin(
        config, repo_root, resume=resume
    )
    total_started = time.perf_counter()
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
    initializer, layout = build_canonical_model(
        model_seed, device=device, dtype=training_dtype
    )
    initialization = _fit_upright_initializer(
        initializer,
        train_dataset,
        test_dataset,
        partitions.initialization,
        partitions.evaluation,
        config,
        nine_prevalence=nine_prevalence,
        device=device,
        dtype=training_dtype,
    )
    initial_state = _state_dict_cpu(initializer)
    initial_state_hash = state_dict_hash(initial_state)
    session.write_json("initialization_metrics.json", initialization)

    dense_fisher, fisher_metrics = _estimate_initial_fisher(
        initializer,
        layout,
        train_dataset,
        partitions.reference,
        config,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
    )
    initial_approximation = approximate_low_rank_diagonal(
        lambda vector: dense_fisher @ vector,
        torch.diagonal(dense_fisher),
        rank=config.fisher.rank,
        seed=derive_component_seed(
            config.replica_seed, "plan5_phase3_initial_lanczos"
        ),
    )
    fisher_metrics["lanczos"] = initial_approximation.diagnostics.mapping()
    fisher_metrics["representation_storage_bytes"] = (
        initial_approximation.representation.storage_bytes()
    )
    selected_reference = tuple(
        partitions.reference[: config.fisher.initial_sample_size]
    )
    session.write_torch(
        "initial_fisher.pt",
        {
            "dense": dense_fisher.cpu(),
            "representation": initial_approximation.representation.artifact_mapping(),
            "sample_indices": torch.as_tensor(selected_reference, dtype=torch.long),
        },
    )

    metrics_by_condition: dict[str, list[dict[str, Any]]] = {}
    trajectories: dict[str, dict[str, Tensor]] = {}
    states: dict[str, dict[str, Tensor]] = {}
    condition_summaries: dict[str, dict[str, Any]] = {}
    for condition in PHASE3_CONDITIONS:
        rows, trajectory, summary, condition_states = _condition_trajectory(
            condition,
            initializer,
            initial_approximation.representation,
            train_dataset,
            test_dataset,
            partitions.evaluation,
            stream_inputs,
            stream_targets,
            stream_plan,
            config,
            nine_prevalence=nine_prevalence,
            device=device,
            training_dtype=training_dtype,
            matrix_dtype=matrix_dtype,
        )
        metrics_by_condition[condition] = rows
        trajectories[condition] = trajectory
        states[condition] = condition_states
        condition_summaries[condition] = summary

    initial_hashes = {
        condition: state_dict_hash(states[condition]["initial"])
        for condition in PHASE3_CONDITIONS
    }
    if set(initial_hashes.values()) != {initial_state_hash}:
        raise RuntimeError("Phase 3 treatments do not share their initializer")
    first_rows = [metrics_by_condition[condition][0] for condition in PHASE3_CONDITIONS]
    for name in (
        "parameter_hash",
        "current_environment_accuracy",
        "current_nll",
        "current_per_class_recall",
    ):
        if first_rows[0][name] != first_rows[1][name]:
            raise RuntimeError(f"Phase 3 pre-treatment pairing failed for {name}")

    session.write_json("trajectory_metrics.json", metrics_by_condition)
    session.write_torch("trajectories.pt", trajectories)
    session.write_torch("model_states.pt", states)
    run_summary = {
        "artifact_schema_version": config.artifact_schema_version,
        "metric_schema_version": config.metric_schema_version,
        "conditions": list(PHASE3_CONDITIONS),
        "schedule_hash": schedule.content_hash,
        "stream_plan_hash": stream_plan.content_hash,
        "partition_hash": partitions.content_hash,
        "initial_model_state_hash": initial_state_hash,
        "condition_initial_model_state_hashes": initial_hashes,
        "parameter_count": layout.total_numel,
        "num_points": schedule.num_points,
        "num_transitions": schedule.num_transitions,
        "samples_per_step": config.data.samples_per_step,
        "nine_prevalence": nine_prevalence,
        "persistent_environment_thresholds": list(
            PERSISTENT_ENVIRONMENT_THRESHOLDS
        ),
        "initial_fisher": fisher_metrics,
        "condition_summaries": condition_summaries,
        "total_wall_time_seconds": time.perf_counter() - total_started,
    }
    session.write_json("run_summary.json", run_summary)
    return session.complete(PHASE3_REQUIRED_ARTIFACTS)


def main() -> None:
    arguments = parse_arguments()
    config = load_phase3_config(arguments.config)
    repo_root = Path(__file__).parents[2]
    path = run_phase3(
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
