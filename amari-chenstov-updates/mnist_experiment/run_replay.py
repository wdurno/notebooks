"""Run immutable Plan 3 pure-replay trajectories with resource accounting."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import resource
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

from mnist_experiment.run_experiment import _materialize_batch, _synchronize
from src.artifacts import RunStore
from src.config import ExperimentConfig, load_config
from src.ewc import build_optimizer, take_ewc_proposal
from src.initialization import evaluate_classifier, load_replica_bundle_for_config
from src.mnist_data import load_mnist_datasets
from src.mnist_model import configure_torch_runtime, resolve_device, resolve_dtype
from src.parameters import ParameterLayout
from src.replay import FifoReplayBuffer, ReplayEvent, stream_events
from src.representations import DiagonalFisher


PLAN3_REPLAY_ARTIFACT_SCHEMA_VERSION = 5
PLAN3_REPLAY_METRIC_SCHEMA_VERSION = 9
CALIBRATION_BINS = 15
_T = TypeVar("_T")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--replica-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_config(config: ExperimentConfig) -> None:
    if config.schema_version != 13:
        raise ValueError("Plan 3 replay runs require schema version 13")
    if config.artifact_schema_version != PLAN3_REPLAY_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Plan 3 replay artifact schema must be version 5")
    if config.metric_schema_version != PLAN3_REPLAY_METRIC_SCHEMA_VERSION:
        raise ValueError("Plan 3 replay metric schema must be version 9")
    if config.replay is None or config.replay.policy != "fifo":
        raise ValueError("Plan 3 replay runs require a FIFO replay configuration")
    if config.controller.policy != "fixed_unified":
        raise ValueError("pure replay requires the fixed unified controller policy")
    if not (
        config.controller.fixed_pi == 1.0
        and config.controller.pi_min == 1.0
        and config.controller.pi_max == 1.0
    ):
        raise ValueError("pure replay requires pi=pi_min=pi_max=1")
    if config.estimator.controller_methods != ["low_rank_diagonal"]:
        raise ValueError("Plan 3 pairing requires the selected representation label")


def _tensor_hash(tensor: Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(str(tuple(cpu.shape)).encode("ascii"))
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_dict_cpu(model: torch.nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _tensor_tree_bytes(value: Any) -> int:
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_tree_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_tree_bytes(item) for item in value)
    return 0


def _timed(
    device: torch.device,
    operation: Callable[[], _T],
) -> tuple[_T, dict[str, float | int | None]]:
    _synchronize(device)
    cuda_start = None
    cuda_end = None
    baseline_cuda = None
    if device.type == "cuda":
        baseline_cuda = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
    started = time.perf_counter()
    result = operation()
    if cuda_end is not None:
        cuda_end.record()
    _synchronize(device)
    wall_seconds = time.perf_counter() - started
    return result, {
        "wall_seconds": wall_seconds,
        "cuda_seconds": (
            None
            if cuda_start is None or cuda_end is None
            else cuda_start.elapsed_time(cuda_end) / 1000.0
        ),
        "peak_cuda_allocated_bytes": (
            None
            if device.type != "cuda"
            else torch.cuda.max_memory_allocated(device)
        ),
        "incremental_peak_cuda_bytes": (
            None
            if baseline_cuda is None
            else max(
                0,
                torch.cuda.max_memory_allocated(device) - baseline_cuda,
            )
        ),
    }


def _active_indices(
    current: Sequence[ReplayEvent],
    replay: Sequence[ReplayEvent],
) -> tuple[int, ...]:
    current_ids = {event.event_id for event in current}
    replay_ids = {event.event_id for event in replay}
    if current_ids.intersection(replay_ids):
        raise RuntimeError("current replay events entered the pre-update buffer")
    return tuple(
        event.observation_index for event in (*current, *replay)
    )


def _operation_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {
            name
            for row in rows
            for name in row.get("operations", {})
        }
    )
    totals = {}
    for name in names:
        operation_rows = [
            row["operations"][name]
            for row in rows
            if name in row.get("operations", {})
        ]
        totals[name] = {
            "wall_seconds": sum(float(item["wall_seconds"]) for item in operation_rows),
            "cuda_seconds": (
                None
                if any(item["cuda_seconds"] is None for item in operation_rows)
                else sum(float(item["cuda_seconds"]) for item in operation_rows)
            ),
            "maximum_peak_cuda_allocated_bytes": (
                None
                if any(
                    item["peak_cuda_allocated_bytes"] is None
                    for item in operation_rows
                )
                else max(
                    int(item["peak_cuda_allocated_bytes"])
                    for item in operation_rows
                )
            ),
            "maximum_incremental_peak_cuda_bytes": (
                None
                if any(
                    item["incremental_peak_cuda_bytes"] is None
                    for item in operation_rows
                )
                else max(
                    int(item["incremental_peak_cuda_bytes"])
                    for item in operation_rows
                )
            ),
        }
    return totals


def _checkpoint_mapping(
    *,
    next_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: FifoReplayBuffer,
    rows: list[dict[str, Any]],
    parameters: list[Tensor],
    displacements: list[Tensor],
    observed_source_indices: set[int],
    cumulative_optimizer_event_evaluations: int,
) -> dict[str, Any]:
    return {
        "schema_version": PLAN3_REPLAY_ARTIFACT_SCHEMA_VERSION,
        "next_step": next_step,
        "model_state": _state_dict_cpu(model),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "replay_state": replay.to_mapping(),
        "rows": rows,
        "parameters": parameters,
        "displacements": displacements,
        "observed_source_indices": sorted(observed_source_indices),
        "cumulative_optimizer_event_evaluations": (
            cumulative_optimizer_event_evaluations
        ),
    }


def main() -> None:
    process_started = time.perf_counter()
    arguments = parse_arguments()
    config = load_config(arguments.config)
    _validate_config(config)
    assert config.replay is not None
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
    output_root = arguments.output_root or Path(config.cache_root)
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.training_dtype)
    session = RunStore(output_root).begin(
        config,
        Path(__file__).parents[1],
        resume=arguments.resume,
    )
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=False)
    loaded = load_replica_bundle_for_config(replica_root, config, device=device)
    model = copy.deepcopy(loaded.model).to(device=device, dtype=training_dtype)
    layout = ParameterLayout.from_module(model)
    optimizer = build_optimizer(model, config.optimizer)
    zero_fisher = DiagonalFisher(
        torch.zeros(layout.total_numel, device=device, dtype=training_dtype)
    )
    capacity = (
        None if config.replay.capacity == "unbounded" else config.replay.capacity
    )
    replay = FifoReplayBuffer(capacity=capacity)
    evaluation_dataset = Subset(test_dataset, loaded.partitions.evaluation)
    effective_steps = config.replay.max_steps or config.data.num_p_steps
    p_values = loaded.stream_plan.p_values[:effective_steps]
    rows: list[dict[str, Any]] = []
    parameters: list[Tensor] = []
    displacements: list[Tensor] = []
    observed_source_indices: set[int] = set()
    cumulative_optimizer_event_evaluations = 0
    start_step = 0
    checkpoint_path = session.path / "plan3_replay_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint.get("schema_version") != PLAN3_REPLAY_ARTIFACT_SCHEMA_VERSION:
            raise RuntimeError("replay checkpoint schema is incompatible")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        replay = FifoReplayBuffer.from_mapping(checkpoint["replay_state"])
        rows = list(checkpoint["rows"])
        parameters = list(checkpoint["parameters"])
        displacements = list(checkpoint["displacements"])
        observed_source_indices = set(checkpoint["observed_source_indices"])
        cumulative_optimizer_event_evaluations = int(
            checkpoint["cumulative_optimizer_event_evaluations"]
        )
        start_step = int(checkpoint["next_step"])
        print(f"resumed Plan 3 replay at step {start_step}", flush=True)

    trajectory_started = time.perf_counter()
    progress = tqdm(
        range(start_step, effective_steps),
        desc=f"replay {config.replay.capacity}",
        unit="step",
    )
    for step in progress:
        p_value = p_values[step]
        parameter_before = layout.flatten_module(model, detach=True)
        parameters.append(parameter_before.cpu())
        classification, evaluation_timing = _timed(
            device,
            lambda: evaluate_classifier(
                model,
                evaluation_dataset,
                batch_size=config.initialization.batch_size,
                device=device,
                dtype=training_dtype,
                num_workers=config.initialization.num_workers,
                calibration_bins=CALIBRATION_BINS,
                nine_prevalence=p_value,
            ),
        )
        operations = {"evaluation": evaluation_timing}
        row: dict[str, Any] = {
            "step": step,
            "p": p_value,
            "parameter_hash": _tensor_hash(parameter_before),
            "classification": classification,
            "operations": operations,
            "proposal": None,
            "replay_before": {
                "event_count": len(replay.events),
                "unique_source_observation_count": len(
                    {event.observation_index for event in replay.events}
                ),
                "event_ids": [event.event_id for event in replay.events],
            },
        }
        if step + 1 < effective_steps:
            current_events = stream_events(
                step,
                loaded.stream_plan.observation_indices[step],
                loaded.stream_plan.class_labels[step],
                samples_per_step=config.data.samples_per_step,
            )
            pre_update_replay = replay.events
            active_indices = _active_indices(current_events, pre_update_replay)
            observed_source_indices.update(
                event.observation_index for event in current_events
            )
            (training_inputs, training_targets), materialization_timing = _timed(
                device,
                lambda: _materialize_batch(
                    train_dataset,
                    active_indices,
                    device=device,
                    dtype=training_dtype,
                ),
            )
            operations["materialization"] = materialization_timing
            expected_targets = torch.as_tensor(
                [
                    event.class_label
                    for event in (*current_events, *pre_update_replay)
                ],
                dtype=torch.long,
                device=device,
            )
            if not torch.equal(training_targets, expected_targets):
                raise RuntimeError("active replay labels do not match the dataset")
            proposal, optimization_timing = _timed(
                device,
                lambda: take_ewc_proposal(
                    model,
                    layout,
                    training_inputs,
                    training_targets,
                    zero_fisher,
                    config.optimizer,
                    optimizer,
                    adaptation_weight=1.0,
                ),
            )
            operations["optimization"] = optimization_timing
            displacement = (
                layout.flatten_module(model, detach=True) - parameter_before
            ).cpu()
            if not torch.equal(displacement, proposal.displacement):
                raise RuntimeError("replay proposal does not equal the realized move")
            displacements.append(displacement)
            evicted, insertion_timing = _timed(
                device,
                lambda: replay.insert(current_events),
            )
            operations["replay_commit"] = insertion_timing
            function_evaluations = proposal.optimizer_function_evaluations
            optimizer_event_evaluations = len(active_indices) * function_evaluations
            cumulative_optimizer_event_evaluations += optimizer_event_evaluations
            row.update(
                {
                    "active_block": {
                        "current_event_count": len(current_events),
                        "replay_event_count": len(pre_update_replay),
                        "presented_event_count": len(active_indices),
                        "unique_event_count": len(
                            {event.event_id for event in (*current_events, *pre_update_replay)}
                        ),
                        "unique_source_observation_count": len(set(active_indices)),
                        "current_event_already_in_replay_count": len(
                            {event.event_id for event in current_events}.intersection(
                                event.event_id for event in pre_update_replay
                            )
                        ),
                    },
                    "proposal": proposal.metrics_mapping(),
                    "optimizer_event_evaluations": optimizer_event_evaluations,
                    "cumulative_optimizer_event_evaluations": (
                        cumulative_optimizer_event_evaluations
                    ),
                    "replay_after": {
                        "event_count": len(replay.events),
                        "unique_source_observation_count": len(
                            {event.observation_index for event in replay.events}
                        ),
                        "inserted_event_count": len(current_events),
                        "evicted_event_count": len(evicted),
                        "total_insertions": replay.total_insertions,
                        "total_evictions": replay.total_evictions,
                        "logical_persistent_bytes": replay.logical_persistent_bytes,
                        "physical_index_state_bytes": (
                            replay.physical_index_state_bytes
                        ),
                        "serialized_state_bytes": replay.serialized_state_bytes,
                    },
                    "cumulative_unique_source_observation_count": len(
                        observed_source_indices
                    ),
                }
            )

        rows.append(row)
        session.write_torch(
            "plan3_replay_checkpoint.pt",
            _checkpoint_mapping(
                next_step=step + 1,
                model=model,
                optimizer=optimizer,
                replay=replay,
                rows=rows,
                parameters=parameters,
                displacements=displacements,
                observed_source_indices=observed_source_indices,
                cumulative_optimizer_event_evaluations=(
                    cumulative_optimizer_event_evaluations
                ),
            ),
        )
    trajectory_elapsed = time.perf_counter() - trajectory_started
    total_elapsed = time.perf_counter() - process_started
    parameter_tensor = torch.stack(parameters)
    displacement_tensor = torch.stack(displacements)
    if not torch.equal(
        displacement_tensor,
        parameter_tensor[1:] - parameter_tensor[:-1],
    ):
        raise RuntimeError("replay displacements do not match parameter states")
    initial_hash = _tensor_hash(parameter_tensor[0])
    trajectory_artifact = {
        "schema_version": PLAN3_REPLAY_ARTIFACT_SCHEMA_VERSION,
        "stream_plan_hash": loaded.stream_plan.content_hash,
        "p_values": p_values,
        "observation_indices": loaded.stream_plan.observation_indices[:effective_steps],
        "class_labels": loaded.stream_plan.class_labels[:effective_steps],
        "parameter_layout": loaded.layout.metadata(),
        "parameters": parameter_tensor,
        "displacements": displacement_tensor,
        "final_replay_state": replay.to_mapping(),
    }
    trajectory_path = session.write_torch(
        "plan3_replay_trajectory.pt",
        trajectory_artifact,
    )
    checkpoint_bytes = checkpoint_path.stat().st_size
    model_parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    metrics = {
        "plan3_replay_metric_schema_version": PLAN3_REPLAY_METRIC_SCHEMA_VERSION,
        "run_kind": "plan3_pure_replay",
        "replica_bundle_id": loaded.metadata["bundle_id"],
        "stream_plan_hash": loaded.stream_plan.content_hash,
        "condition": {
            "data_mode": "replay" if capacity != 0 else "current",
            "replay_policy": "fifo",
            "replay_capacity": config.replay.capacity,
            "uses_ewc": False,
            "adaptation_weight": 1.0,
            "active_likelihood": "current_batch_plus_preupdate_replay_buffer",
            "stream_sampling": "with_replacement",
            "duplicate_source_observations": "retained_as_distinct_events",
            "insertion_transaction": "post_successful_step",
        },
        "steps": effective_steps,
        "optimizer_steps": effective_steps - 1,
        "parameter_count": layout.total_numel,
        "initial_parameter_hash": initial_hash,
        "condition_steps": rows,
        "resource_ledger": {
            "operation_totals": _operation_totals(rows),
            "total_wall_seconds": total_elapsed,
            "trajectory_wall_seconds": trajectory_elapsed,
            "optimizer_iterations": sum(
                int(row["proposal"]["optimizer_iterations"])
                for row in rows
                if row["proposal"] is not None
            ),
            "optimizer_function_evaluations": sum(
                int(row["proposal"]["optimizer_function_evaluations"])
                for row in rows
                if row["proposal"] is not None
            ),
            "optimizer_event_evaluations": (
                cumulative_optimizer_event_evaluations
            ),
            "logical_replay_persistent_bytes_final": (
                replay.logical_persistent_bytes
            ),
            "physical_replay_index_state_bytes_final": (
                replay.physical_index_state_bytes
            ),
            "serialized_replay_state_bytes_final": replay.serialized_state_bytes,
            "common_model_parameter_bytes": model_parameter_bytes,
            "common_optimizer_tensor_bytes_final": _tensor_tree_bytes(
                optimizer.state_dict()
            ),
            "peak_process_rss_bytes": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "peak_cuda_memory_bytes": (
                None
                if device.type != "cuda"
                else max(
                    int(item["peak_cuda_allocated_bytes"])
                    for row in rows
                    for item in row["operations"].values()
                    if item["peak_cuda_allocated_bytes"] is not None
                )
            ),
            "checkpoint_file_bytes": checkpoint_bytes,
        },
        "pairing": {
            "shared_initialization": True,
            "shared_observation_stream": True,
            "replica_bundle_id": loaded.metadata["bundle_id"],
            "initial_parameter_hash": initial_hash,
        },
        "classification_contract": {
            "nine_recall": "true_positive_rate_conditioned_on_true_nine",
            "nine_false_positive_rate": "predicted_nine_conditioned_on_true_non_nine",
            "nine_ovr_accuracy": "prevalence_adjusted_at_row_p",
            "nine_precision": "prevalence_adjusted_at_row_p_null_when_undefined",
            "environment_accuracy": "multiclass_prevalence_adjusted_at_row_p",
        },
        "artifact_files_bytes": {
            "plan3_replay_trajectory.pt": trajectory_path.stat().st_size,
            "plan3_replay_checkpoint.pt": checkpoint_bytes,
        },
    }
    session.write_json("plan3_replay_metrics.json", metrics)
    destination = session.complete(
        [
            "plan3_replay_metrics.json",
            "plan3_replay_trajectory.pt",
            "plan3_replay_checkpoint.pt",
        ]
    )
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "path": str(destination),
                "capacity": config.replay.capacity,
                "steps": effective_steps,
                "wall_seconds": total_elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
