"""Run the immutable Plan 5 Phase 2 learnability and geometry audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm.auto import tqdm

from src.initialization import evaluate_classifier, state_dict_hash
from src.mnist_data import ReferenceSamplePlan, dataset_targets, load_mnist_datasets
from src.mnist_model import (
    build_canonical_model,
    configure_torch_runtime,
    mnist_nll,
    resolve_device,
    resolve_dtype,
)
from src.parameters import ParameterLayout
from src.reference import chunked_reference_fisher
from src.seeding import derive_component_seed

from .artifacts import AUDIT_REQUIRED_ARTIFACTS, RotatedAuditRunStore
from .audit import (
    feasibility_gate,
    fisher_summary,
    offline_risk_coefficients,
    reference_path_diagnostics,
    repeated_fit_noise,
)
from .audit_config import ReferenceFitConfig, RotatedAuditConfig, load_audit_config
from .data import partition_all_digit_mnist
from .run import CALIBRATION_BINS, _fit_upright_initializer
from .transform import rotate_mnist_batch, rotate_mnist_tensor, tensor_content_hash


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--data-root", type=Path, default=Path("cache/mnist_experiment/datasets")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/mnist_experiment/rotated_mnist/audits"),
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _state_dict_cpu(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _materialize_rotated(
    dataset: Dataset,
    indices: Sequence[int],
    *,
    angle_degrees: float,
    config: RotatedAuditConfig,
) -> tuple[TensorDataset, str]:
    loader = DataLoader(
        Subset(dataset, tuple(indices)),
        batch_size=1024,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        persistent_workers=config.runtime.num_workers > 0,
    )
    inputs = []
    targets = []
    for batch_inputs, batch_targets in loader:
        inputs.append(
            rotate_mnist_batch(batch_inputs, angle_degrees, config.rotation)
        )
        targets.append(batch_targets.to(device="cpu", dtype=torch.long))
    input_tensor = torch.cat(inputs).contiguous()
    target_tensor = torch.cat(targets).contiguous()
    digest = hashlib.sha256(
        (
            tensor_content_hash(input_tensor)
            + tensor_content_hash(target_tensor)
        ).encode("ascii")
    ).hexdigest()
    return TensorDataset(input_tensor, target_tensor), digest


def _select_positions(pool_size: int, sample_size: int, seed: int) -> tuple[int, ...]:
    order = torch.randperm(
        pool_size, generator=torch.Generator().manual_seed(seed)
    )
    return tuple(order[:sample_size].tolist())


def _build_optimizer(
    model: nn.Module, config: ReferenceFitConfig
) -> torch.optim.Optimizer:
    if config.optimizer != "adam":
        raise ValueError("unsupported reference optimizer")
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _fit_reference(
    model: nn.Module,
    training_dataset: Dataset,
    validation_dataset: Dataset,
    config: RotatedAuditConfig,
    *,
    loader_seed: int,
    nine_prevalence: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    loader = DataLoader(
        training_dataset,
        batch_size=config.reference.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(loader_seed),
        num_workers=config.runtime.num_workers,
        persistent_workers=config.runtime.num_workers > 0,
    )
    optimizer = _build_optimizer(model, config.reference)
    history = []
    best_nll = math.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, config.reference.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=dtype)
            targets = targets.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * targets.numel()
            total_count += targets.numel()
        validation = evaluate_classifier(
            model,
            validation_dataset,
            batch_size=config.reference.batch_size,
            device=device,
            dtype=dtype,
            num_workers=config.runtime.num_workers,
            calibration_bins=CALIBRATION_BINS,
            nine_prevalence=nine_prevalence,
        )
        validation_nll = float(validation["nll"])
        history.append(
            {
                "epoch": epoch,
                "training_nll": total_loss / total_count,
                "validation_nll": validation_nll,
                "validation_environment_accuracy": validation[
                    "environment_accuracy"
                ],
                "validation_nine_ovr_accuracy": validation[
                    "nine_ovr_accuracy"
                ],
            }
        )
        if validation_nll < best_nll - config.reference.minimum_delta:
            best_nll = validation_nll
            best_epoch = epoch
            best_state = _state_dict_cpu(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.reference.patience:
                break
    if best_state is None:
        raise RuntimeError("reference optimization produced no finite best state")
    model.load_state_dict(best_state)
    return {
        "wall_time_seconds": time.perf_counter() - started,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_nll": best_nll,
        "stopped_on_patience": len(history) < config.reference.max_epochs,
        "history": history,
    }


def _confusion_matrix(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[list[int]]:
    result = torch.zeros((10, 10), dtype=torch.long)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            predictions = model(inputs.to(device=device, dtype=dtype)).argmax(dim=1)
            flat = targets.to(device=predictions.device) * 10 + predictions
            result += torch.bincount(flat.cpu(), minlength=100).reshape(10, 10)
    if was_training:
        model.train()
    return result.tolist()


def _reference_plan(
    config: RotatedAuditConfig,
    *,
    partitions: Any,
    primary_positions: tuple[int, ...],
    repeat_positions: tuple[tuple[int, ...], ...],
    fit_candidate_indices: tuple[int, ...],
    fisher_indices: tuple[int, ...],
    validation_indices: tuple[int, ...],
    reporting_indices: tuple[int, ...],
) -> dict[str, Any]:
    def source_indices(positions: tuple[int, ...]) -> list[int]:
        return [fit_candidate_indices[position] for position in positions]

    return {
        "schema_version": 1,
        "angles_degrees": list(config.reference.angles_degrees),
        "partition_hash": partitions.content_hash,
        "fit_candidate_indices": list(fit_candidate_indices),
        "primary_fit_indices": source_indices(primary_positions),
        "endpoint_repeat_fit_indices": [
            source_indices(positions) for positions in repeat_positions
        ],
        "fisher_indices": list(fisher_indices),
        "validation_indices": list(validation_indices),
        "reporting_indices": list(reporting_indices),
        "pairing": (
            "primary source observations and optimizer seed are matched across angles; "
            "repeat index is matched across endpoints"
        ),
        "fisher_role": "held out from every reference-model fit",
        "reporting_role": "held out from fitting and early stopping",
    }


def _example_panel(
    dataset: Dataset,
    targets: Tensor,
    reporting_indices: tuple[int, ...],
    angles: tuple[float, ...],
    config: RotatedAuditConfig,
) -> tuple[dict[str, Tensor], list[dict[str, Any]]]:
    selected = []
    for label in range(10):
        selected.append(next(index for index in reporting_indices if int(targets[index]) == label))
    panel = []
    rows = []
    baseline_mass = None
    for angle in angles:
        images = torch.stack(
            [rotate_mnist_tensor(dataset[index][0], angle, config.rotation) for index in selected]
        )
        mass = float(images.abs().sum())
        if baseline_mass is None:
            baseline_mass = mass
        border = torch.zeros_like(images, dtype=torch.bool)
        border[..., :2, :] = True
        border[..., -2:, :] = True
        border[..., :, :2] = True
        border[..., :, -2:] = True
        rows.append(
            {
                "angle_degrees": angle,
                "absolute_intensity_mass": mass,
                "intensity_mass_ratio_to_upright": mass / max(baseline_mass, 1e-12),
                "border_absolute_mass_fraction": float(images.abs()[border].sum())
                / max(mass, 1e-12),
                "nonzero_pixel_fraction": float((images.abs() > 1e-8).double().mean()),
                "content_hash": tensor_content_hash(images),
            }
        )
        panel.append(images)
    return {
        "angles_degrees": torch.tensor(angles, dtype=torch.float64),
        "labels": torch.arange(10, dtype=torch.long),
        "source_indices": torch.tensor(selected, dtype=torch.long),
        "images": torch.stack(panel),
    }, rows


def run_audit(
    config: RotatedAuditConfig,
    *,
    data_root: str | Path,
    output_root: str | Path,
    repo_root: str | Path,
    download: bool = False,
    resume: bool = False,
) -> Path:
    config.validate()
    device = resolve_device(config.runtime.device)
    dtype = resolve_dtype(config.runtime.dtype)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=device.type == "cuda",
    )
    session = RotatedAuditRunStore(output_root).begin(
        config, repo_root, resume=resume
    )
    started = time.perf_counter()
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
    session.write_json("partitions.json", partitions.to_mapping())

    reference_indices = partitions.reference
    fit_candidates = tuple(reference_indices[: config.reference.fit_pool_size])
    fisher_indices = tuple(reference_indices[config.reference.fit_pool_size :])
    validation_indices = tuple(
        partitions.evaluation[: config.reference.validation_size]
    )
    reporting_indices = tuple(
        partitions.evaluation[config.reference.validation_size :]
    )
    primary_seed = derive_component_seed(
        config.replica_seed, "plan5_audit_primary_fit"
    )
    repeat_seed = derive_component_seed(
        config.replica_seed, "plan5_audit_endpoint_repeats"
    )
    primary_positions = _select_positions(
        len(fit_candidates), config.reference.fit_sample_size, primary_seed
    )
    repeat_positions = tuple(
        _select_positions(
            len(fit_candidates),
            config.reference.fit_sample_size,
            repeat_seed + repeat,
        )
        for repeat in range(config.reference.endpoint_repeat_count)
    )
    plan = _reference_plan(
        config,
        partitions=partitions,
        primary_positions=primary_positions,
        repeat_positions=repeat_positions,
        fit_candidate_indices=fit_candidates,
        fisher_indices=fisher_indices,
        validation_indices=validation_indices,
        reporting_indices=reporting_indices,
    )
    plan["content_hash"] = _canonical_hash(plan)
    session.write_json("reference_plan.json", plan)

    model_seed = derive_component_seed(
        config.replica_seed, "plan5_model_initialization"
    )
    initializer, layout = build_canonical_model(model_seed, device=device, dtype=dtype)
    initialization = _fit_upright_initializer(
        initializer,
        train_dataset,
        test_dataset,
        partitions.initialization,
        validation_indices,
        config,
        nine_prevalence=nine_prevalence,
        device=device,
        dtype=dtype,
    )
    initial_state = _state_dict_cpu(initializer)
    initial_state_hash = state_dict_hash(initial_state)
    session.write_json("initialization_metrics.json", initialization)

    examples, transform_diagnostics = _example_panel(
        test_dataset,
        test_targets,
        reporting_indices,
        config.reference.angles_degrees,
        config,
    )
    session.write_torch("transform_examples.pt", examples)

    zero_shot_rows = []
    reference_rows = []
    states: dict[str, dict[str, Tensor]] = {"initializer": initial_state}
    fishers: dict[str, Tensor] = {}
    parameters: dict[str, Tensor] = {}
    materialization_rows = []
    endpoint_fits: dict[float, list[tuple[str, Tensor, Tensor]]] = {
        config.reference.angles_degrees[0]: [],
        config.reference.angles_degrees[-1]: [],
    }

    progress = tqdm(
        config.reference.angles_degrees,
        desc="phase2 reference angles",
        unit="angle",
    )
    for angle in progress:
        fit_pool, fit_hash = _materialize_rotated(
            train_dataset, fit_candidates, angle_degrees=angle, config=config
        )
        fisher_dataset, fisher_hash = _materialize_rotated(
            train_dataset, fisher_indices, angle_degrees=angle, config=config
        )
        validation_dataset, validation_hash = _materialize_rotated(
            test_dataset, validation_indices, angle_degrees=angle, config=config
        )
        reporting_dataset, reporting_hash = _materialize_rotated(
            test_dataset, reporting_indices, angle_degrees=angle, config=config
        )
        materialization_rows.append(
            {
                "angle_degrees": angle,
                "fit_pool_hash": fit_hash,
                "fisher_hash": fisher_hash,
                "validation_hash": validation_hash,
                "reporting_hash": reporting_hash,
            }
        )
        zero_metrics = evaluate_classifier(
            initializer,
            reporting_dataset,
            batch_size=config.reference.batch_size,
            device=device,
            dtype=dtype,
            num_workers=config.runtime.num_workers,
            calibration_bins=CALIBRATION_BINS,
            nine_prevalence=nine_prevalence,
        )
        zero_shot_rows.append({"angle_degrees": angle, **zero_metrics})

        roles: list[tuple[str, tuple[int, ...], int]] = [
            ("primary", primary_positions, primary_seed)
        ]
        if angle in endpoint_fits:
            roles.extend(
                (
                    f"endpoint_repeat_{repeat + 1}",
                    positions,
                    repeat_seed + repeat,
                )
                for repeat, positions in enumerate(repeat_positions)
            )
        for role, positions, loader_seed in roles:
            fit_id = f"angle-{int(round(angle * 10)):04d}__{role}"
            model, fit_layout = build_canonical_model(
                model_seed, device=device, dtype=dtype
            )
            fit_layout.assert_metadata(layout.metadata())
            model.load_state_dict(initial_state)
            fit_metrics = _fit_reference(
                model,
                Subset(fit_pool, positions),
                validation_dataset,
                config,
                loader_seed=loader_seed,
                nine_prevalence=nine_prevalence,
                device=device,
                dtype=dtype,
            )
            evaluation = evaluate_classifier(
                model,
                reporting_dataset,
                batch_size=config.reference.batch_size,
                device=device,
                dtype=dtype,
                num_workers=config.runtime.num_workers,
                calibration_bins=CALIBRATION_BINS,
                nine_prevalence=nine_prevalence,
            )
            fisher_plan = ReferenceSamplePlan(
                p=nine_prevalence,
                observation_indices=tuple(range(len(fisher_dataset))),
                class_labels=tuple(
                    int(value) for value in fisher_dataset.tensors[1].tolist()
                ),
                non_nine_sampling="empirical",
                seed=derive_component_seed(
                    config.replica_seed, f"plan5_audit_fisher_{fit_id}"
                ),
                partition_hash=partitions.content_hash,
                pool="reference",
            )
            fisher_estimate = chunked_reference_fisher(
                model,
                fisher_dataset,
                fisher_plan,
                mnist_nll,
                fit_layout,
                chunk_size=config.reference.fisher_chunk_size,
                device=device,
                derivative_dtype=dtype,
                matrix_dtype=torch.float64,
                strategy="vmap",
                num_workers=config.runtime.num_workers,
            )
            state = _state_dict_cpu(model)
            parameter = fit_layout.flatten_module(model, detach=True).cpu()
            fisher = fisher_estimate.matrix
            states[fit_id] = state
            parameters[fit_id] = parameter
            fishers[fit_id] = fisher
            confusion = (
                _confusion_matrix(
                    model,
                    reporting_dataset,
                    batch_size=config.reference.batch_size,
                    device=device,
                    dtype=dtype,
                )
                if role == "primary"
                else None
            )
            row = {
                "fit_id": fit_id,
                "role": role,
                "angle_degrees": angle,
                "fit_sample_size": len(positions),
                "fit_source_hash": _canonical_hash(
                    [fit_candidates[position] for position in positions]
                ),
                "initial_state_hash": initial_state_hash,
                "model_state_hash": state_dict_hash(state),
                "parameter_hash": tensor_content_hash(parameter),
                "fisher_hash": tensor_content_hash(fisher),
                "optimization": fit_metrics,
                "evaluation": evaluation,
                "confusion_matrix": confusion,
                "fisher_estimation": fisher_estimate.diagnostics_mapping(),
                "fisher_summary": fisher_summary(fisher),
            }
            reference_rows.append(row)
            if angle in endpoint_fits:
                endpoint_fits[angle].append((fit_id, parameter, fisher))
            progress.set_postfix_str(
                f"fit={fit_id}, acc={float(evaluation['environment_accuracy']):.3f}"
            )

    primary_rows = [row for row in reference_rows if row["role"] == "primary"]
    primary_rows.sort(key=lambda row: row["angle_degrees"])
    primary_parameters = [parameters[row["fit_id"]] for row in primary_rows]
    primary_fishers = [fishers[row["fit_id"]] for row in primary_rows]
    path_rows = reference_path_diagnostics(
        [row["angle_degrees"] for row in primary_rows],
        primary_parameters,
        primary_fishers,
    )
    noise_rows = repeated_fit_noise(endpoint_fits)
    edr_rows = offline_risk_coefficients(
        path_rows,
        parameter_count=layout.total_numel,
        initialization_sample_size=config.data.initialization_size,
        batch_size=config.reference.edr_batch_size,
        fixed_pi=config.reference.edr_fixed_pi,
    )
    zero_30 = next(
        row for row in zero_shot_rows if row["angle_degrees"] == 30.0
    )
    reference_30 = next(
        row for row in primary_rows if row["angle_degrees"] == 30.0
    )
    gate = feasibility_gate(
        zero_shot_environment_accuracy_30=float(zero_30["environment_accuracy"]),
        reference_environment_accuracy_30=float(
            reference_30["evaluation"]["environment_accuracy"]
        ),
        path_rows=path_rows,
        noise_rows=noise_rows,
        config=config.gate,
    )
    path_artifact = {
        "path": path_rows,
        "repeated_fit_noise": noise_rows,
        "offline_risk_coefficients": edr_rows,
        "transform_diagnostics": transform_diagnostics,
        "materialization": materialization_rows,
        "gate": gate,
    }
    elapsed = time.perf_counter() - started
    session.write_json("zero_shot_metrics.json", zero_shot_rows)
    session.write_json("reference_metrics.json", reference_rows)
    session.write_json("path_metrics.json", path_artifact)
    session.write_torch(
        "reference_states.pt",
        {"states": states, "parameters": parameters, "layout": layout.metadata()},
    )
    session.write_torch("reference_fishers.pt", fishers)
    session.write_json(
        "audit_summary.json",
        {
            "artifact_schema_version": config.artifact_schema_version,
            "metric_schema_version": config.metric_schema_version,
            "run_kind": "phase2_learnability_audit",
            "parameter_count": layout.total_numel,
            "initializer_state_hash": initial_state_hash,
            "partition_hash": partitions.content_hash,
            "reference_plan_hash": plan["content_hash"],
            "reference_fit_count": len(reference_rows),
            "primary_angle_count": len(primary_rows),
            "fisher_sample_size": len(fisher_indices),
            "reporting_sample_size": len(reporting_indices),
            "wall_time_seconds": elapsed,
            "gate": gate,
        },
    )
    return session.complete(required=AUDIT_REQUIRED_ARTIFACTS)


def main() -> None:
    arguments = parse_arguments()
    config = load_audit_config(arguments.config)
    repo_root = Path(__file__).parents[2]
    path = run_audit(
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
