"""Calibrate the local EWC optimizer budget from immutable Phase 8 states."""

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
from tqdm.auto import tqdm

from mnist_experiment.run_experiment import (
    _cpu_tree,
    _materialize_batch,
    _synchronize,
)
from src.artifacts import RunStore
from src.config import load_config
from src.ewc import build_optimizer, take_ewc_proposal
from src.fit_calibration import (
    displacement_comparison,
    load_fit_calibration_config,
)
from src.initialization import load_replica_bundle_for_config
from src.mnist_data import load_mnist_datasets
from src.mnist_model import configure_torch_runtime, resolve_device, resolve_dtype
from src.parameters import ParameterLayout
from src.representations import DenseFisher, representation_from_artifact

SOURCE_METRIC_SCHEMA_VERSION = 5
SOURCE_ARTIFACT_SCHEMA_VERSION = 4


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--replica-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"artifact must be an object: {path}")
    return value


def _load_torch(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise RuntimeError(f"artifact must be a mapping: {path}")
    return value


def _validate_source(
    source_run: Path,
    calibration_config,
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not (source_run / "COMPLETED").is_file():
        raise RuntimeError(f"source Phase 8 run is incomplete: {source_run}")
    source_config = load_config(source_run / "config.json")
    if (
        source_config.replica_id != calibration_config.replica_id
        or source_config.replica_seed != calibration_config.replica_seed
    ):
        raise RuntimeError("calibration and source replica identities differ")
    if (
        source_config.runtime.training_dtype
        != calibration_config.runtime.training_dtype
    ):
        raise RuntimeError("calibration must preserve the source training dtype")
    if (
        source_config.runtime.matrix_dtype
        != calibration_config.runtime.matrix_dtype
    ):
        raise RuntimeError("calibration must preserve the source matrix dtype")
    if calibration_config.optimizer is None and (
        source_config.optimizer.inner_steps
        not in calibration_config.inner_step_budgets
    ):
        raise RuntimeError(
            "calibration budgets must include the source inner-step count"
        )
    if (
        calibration_config.optimizer is None
        and source_config.optimizer.name != "sgd"
    ):
        raise RuntimeError(
            "legacy independent budget restarts require stateless SGD"
        )
    if calibration_config.optimizer is not None and (
        calibration_config.optimizer.ewc_strength
        != source_config.optimizer.ewc_strength
    ):
        raise RuntimeError("calibration must preserve the source EWC strength")


    metrics = _load_json(source_run / "phase8_metrics.json")
    if metrics.get("phase8_metric_schema_version") != SOURCE_METRIC_SCHEMA_VERSION:
        raise RuntimeError("fit calibration requires corrected Phase 8 metrics")
    checkpoints = _load_torch(source_run / "phase8_checkpoints.pt")
    trajectories = _load_torch(source_run / "phase8_trajectories.pt")
    if (
        checkpoints.get("schema_version") != SOURCE_ARTIFACT_SCHEMA_VERSION
        or trajectories.get("schema_version") != SOURCE_ARTIFACT_SCHEMA_VERSION
    ):
        raise RuntimeError("fit calibration requires corrected Phase 8 artifacts")
    if metrics.get("stream_plan_hash") != trajectories.get("stream_plan_hash"):
        raise RuntimeError("source metric and trajectory stream hashes differ")
    return source_config, metrics, checkpoints, trajectories


def _fisher_from_checkpoint(
    value: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    representation = representation_from_artifact(value, device=device).to(
        dtype=dtype
    )
    if isinstance(representation, DenseFisher):
        return representation.matrix
    return representation


def main() -> None:
    arguments = parse_arguments()
    config = load_fit_calibration_config(arguments.config)
    source_run = Path(config.source_run)
    source_config, source_metrics, checkpoints, trajectories = _validate_source(
        source_run,
        config,
    )
    optimizer_template = config.optimizer or source_config.optimizer
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=(
            config.runtime.deterministic_algorithms
            and config.runtime.device in {"cuda", "auto"}
        ),
    )
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.training_dtype)
    cache_parent = Path(source_config.cache_root).parent
    data_root = arguments.data_root or cache_parent / "datasets"
    replica_root = arguments.replica_root or cache_parent / "replicas"
    output_root = arguments.output_root or Path(config.cache_root)
    session = RunStore(output_root).begin(
        config,
        Path(__file__).parents[1],
        resume=arguments.resume,
    )

    train_dataset, _ = load_mnist_datasets(data_root, download=False)
    loaded = load_replica_bundle_for_config(
        replica_root,
        source_config,
        device=device,
    )
    loaded.layout.assert_metadata(trajectories["parameter_layout"])
    if loaded.stream_plan.content_hash != trajectories["stream_plan_hash"]:
        raise RuntimeError("source trajectory and replica stream plans differ")
    if tuple(loaded.stream_plan.p_values) != tuple(trajectories["p_values"]):
        raise RuntimeError("source trajectory and replica p grids differ")

    methods = tuple(str(value) for value in source_metrics["methods"])
    total_trials = (
        len(methods)
        * len(config.checkpoint_steps)
        * len(config.inner_step_budgets)
    )
    rows: list[dict[str, Any]] = []
    displacement_artifact: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    progress = tqdm(total=total_trials, desc="EWC fit calibration", unit="fit")

    for method in methods:
        method_checkpoints = checkpoints["conditions"].get(method)
        if not isinstance(method_checkpoints, dict):
            raise RuntimeError(f"source checkpoints omit method {method}")
        displacement_artifact[method] = {}
        for step in config.checkpoint_steps:
            checkpoint = method_checkpoints.get(str(step))
            if not isinstance(checkpoint, dict):
                raise RuntimeError(
                    f"source checkpoints omit {method} step {step}"
                )
            parameter = checkpoint["parameter"].to(
                device=device,
                dtype=training_dtype,
            )
            batch_indices = trajectories["observation_indices"][step]
            inputs, targets = _materialize_batch(
                train_dataset,
                batch_indices,
                device=device,
                dtype=training_dtype,
            )
            expected_targets = torch.as_tensor(
                trajectories["class_labels"][step],
                dtype=torch.long,
                device=device,
            )
            if not torch.equal(targets, expected_targets):
                raise RuntimeError("calibration batch labels do not match source")
            fisher = _fisher_from_checkpoint(
                checkpoint["representation"],
                device=device,
                dtype=training_dtype,
            )
            pi = float(checkpoint["controller_decision"]["applied_pi"])
            group_rows = []
            group_displacements: dict[int, torch.Tensor] = {}

            for budget in config.inner_step_budgets:
                model = copy.deepcopy(loaded.model).to(
                    device=device,
                    dtype=training_dtype,
                )
                layout = ParameterLayout.from_module(model)
                layout.copy_vector_to_module(model, parameter)
                optimizer_config = dataclasses.replace(
                    optimizer_template,
                    inner_steps=budget,
                )
                _synchronize(device)
                trial_started = time.perf_counter()
                proposal = take_ewc_proposal(
                    model,
                    layout,
                    inputs,
                    targets,
                    fisher,
                    optimizer_config,
                    build_optimizer(model, optimizer_config),
                    adaptation_weight=pi,
                )
                _synchronize(device)
                elapsed = time.perf_counter() - trial_started
                displacement = proposal.displacement.to(dtype=torch.float64)
                group_displacements[budget] = displacement
                row = {
                    "method": method,
                    "step": step,
                    "p": float(trajectories["p_values"][step]),
                    "pi": pi,
                    "inner_steps": budget,
                    "elapsed_seconds": elapsed,
                    **proposal.metrics_mapping(),
                }
                source_displacement = checkpoint.get("accepted_displacement")
                if source_displacement is not None:
                    row["comparison_to_source_displacement"] = (
                        displacement_comparison(
                            displacement,
                            source_displacement.to(dtype=torch.float64),
                        )
                    )
                else:
                    row["comparison_to_source_displacement"] = None
                rows.append(row)
                group_rows.append(row)
                progress.update(1)

            maximum_budget = config.inner_step_budgets[-1]
            reference_displacement = group_displacements[maximum_budget]
            reference_row = next(
                row for row in group_rows if row["inner_steps"] == maximum_budget
            )
            previous_displacement = None
            for row in group_rows:
                budget = int(row["inner_steps"])
                displacement = group_displacements[budget]
                row["comparison_to_maximum_budget"] = displacement_comparison(
                    displacement,
                    reference_displacement,
                )
                row["objective_gap_to_maximum_budget"] = (
                    float(row["objective_after"])
                    - float(reference_row["objective_after"])
                )
                row["comparison_to_previous_budget"] = (
                    None
                    if previous_displacement is None
                    else displacement_comparison(
                        displacement,
                        previous_displacement,
                    )
                )
                previous_displacement = displacement
            displacement_artifact[method][str(step)] = {
                str(budget): value
                for budget, value in group_displacements.items()
            }
    progress.close()

    metrics = {
        "fit_calibration_metric_schema_version": config.metric_schema_version,
        "source_run": str(source_run),
        "source_run_id": source_run.name,
        "source_config_hash": source_config.config_hash,
        "source_metric_schema_version": source_metrics[
            "phase8_metric_schema_version"
        ],
        "source_oracle_path_hash": source_metrics["oracle_path_hash"],
        "stream_plan_hash": trajectories["stream_plan_hash"],
        "methods": list(methods),
        "optimizer": dataclasses.asdict(optimizer_template),
        "checkpoint_steps": config.checkpoint_steps,
        "inner_step_budgets": config.inner_step_budgets,
        "restart_contract": "same_anchor_batch_pi_and_fisher_per_budget",
        "maximum_budget_role": "numerical_reference_not_assumed_optimum",
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None
        ),
        "peak_process_rss_bytes": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
    }
    displacement_path = session.write_torch(
        "fit_calibration_displacements.pt",
        _cpu_tree(
            {
                "schema_version": config.artifact_schema_version,
                "source_run_id": source_run.name,
                "displacements": displacement_artifact,
            }
        ),
    )
    metrics["artifact_files_bytes"] = {
        "fit_calibration_displacements.pt": displacement_path.stat().st_size,
    }
    session.write_json("fit_calibration_metrics.json", metrics)
    destination = session.complete(
        ["fit_calibration_metrics.json", "fit_calibration_displacements.pt"]
    )
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "path": str(destination),
                "source_run_id": source_run.name,
                "methods": list(methods),
                "checkpoint_steps": config.checkpoint_steps,
                "inner_step_budgets": config.inner_step_budgets,
                "fits": total_trials,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
