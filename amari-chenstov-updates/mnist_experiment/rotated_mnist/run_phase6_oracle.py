"""Run checkpointed Plan 6 local-oracle estimation."""

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

from src.derivatives import per_sample_derivatives
from src.initialization import state_dict_hash
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import (
    build_canonical_model,
    configure_torch_runtime,
    mnist_nll,
    resolve_device,
    resolve_dtype,
)
from src.parameters import ParameterLayout
from src.representations import representation_from_artifact
from src.seeding import derive_component_seed

from .artifacts import _read_json
from .phase5_single_lap_artifacts import load_completed_single_lap_run
from .phase6_artifacts import PHASE6_ORACLE_REQUIRED_ARTIFACTS, Phase6RunStore
from .phase6_config import Phase6OracleConfig, load_phase6_oracle_config
from .phase6_oracle import (
    angle_key,
    approximate_score_fisher,
    bootstrap_oracle_pi,
    canonical_angle,
    estimate_oracle,
    full_reference_angle_union,
    required_angles,
    transition_steps,
)
from .transform import rotate_mnist_batch, tensor_content_hash


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--data-root", type=Path, default=Path("cache/mnist_experiment/datasets")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("cache/mnist_experiment/rotated_mnist/phase6/oracle"),
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
    angle: float,
    config: Phase6OracleConfig,
) -> TensorDataset:
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
        inputs.append(rotate_mnist_batch(batch_inputs, angle, config_source_rotation(config)))
        targets.append(batch_targets.to(device="cpu", dtype=torch.long))
    return TensorDataset(torch.cat(inputs).contiguous(), torch.cat(targets).contiguous())


_SOURCE_ROTATION = None


def config_source_rotation(config: Phase6OracleConfig):
    if _SOURCE_ROTATION is None:
        raise RuntimeError("source rotation contract has not been initialized")
    return _SOURCE_ROTATION


def _mean_nll(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = 0.0
    count = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=dtype)
            targets = targets.to(device=device)
            loss = nn.functional.cross_entropy(model(inputs), targets, reduction="sum")
            total += float(loss)
            count += targets.numel()
    if was_training:
        model.train()
    return total / count


def _fit_reference(
    model: nn.Module,
    training: Dataset,
    validation: Dataset,
    config: Phase6OracleConfig,
    *,
    seed: int,
    max_epochs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    reference = config.reference
    loader = DataLoader(
        training,
        batch_size=reference.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=config.runtime.num_workers,
        persistent_workers=config.runtime.num_workers > 0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=reference.learning_rate,
        weight_decay=reference.weight_decay,
    )
    best_nll = _mean_nll(
        model,
        validation,
        batch_size=reference.batch_size,
        device=device,
        dtype=dtype,
    )
    initial_nll = best_nll
    best_epoch = 0
    best_state = _state_dict_cpu(model)
    history = []
    without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        training_total = 0.0
        training_count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=dtype)
            targets = targets.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            training_total += float(loss.detach()) * targets.numel()
            training_count += targets.numel()
        validation_nll = _mean_nll(
            model,
            validation,
            batch_size=reference.batch_size,
            device=device,
            dtype=dtype,
        )
        history.append(
            {
                "epoch": epoch,
                "training_nll": training_total / training_count,
                "validation_nll": validation_nll,
            }
        )
        if validation_nll < best_nll - reference.minimum_delta:
            best_nll = validation_nll
            best_epoch = epoch
            best_state = _state_dict_cpu(model)
            without_improvement = 0
        else:
            without_improvement += 1
            if without_improvement >= reference.patience:
                break
    model.load_state_dict(best_state)
    return {
        "initial_validation_nll": initial_nll,
        "best_validation_nll": best_nll,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "epoch_budget": max_epochs,
        "stopped_on_patience": len(history) < max_epochs,
        "wall_time_seconds": time.perf_counter() - started,
        "history": history,
    }


def _fit_local_mle(
    model: nn.Module,
    dataset: Dataset,
    config: Phase6OracleConfig,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    local = config.local_mle
    loader = DataLoader(
        dataset,
        batch_size=local.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=local.learning_rate, weight_decay=local.weight_decay
    )
    def prediction_snapshot() -> tuple[float, float, Tensor]:
        evaluation_loader = DataLoader(
            dataset, batch_size=local.batch_size, shuffle=False, num_workers=0
        )
        log_probabilities = []
        total_nll = 0.0
        correct = 0
        count = 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in evaluation_loader:
                inputs = inputs.to(device=device, dtype=dtype)
                targets = targets.to(device=device)
                log_probs = nn.functional.log_softmax(model(inputs), dim=1)
                total_nll += float(
                    nn.functional.nll_loss(log_probs, targets, reduction="sum")
                )
                correct += int((log_probs.argmax(dim=1) == targets).sum())
                count += targets.numel()
                log_probabilities.append(log_probs.detach().cpu().to(torch.float64))
        return total_nll / count, correct / count, torch.cat(log_probabilities)

    initial_nll, initial_accuracy, reference_log_probs = prediction_snapshot()
    started = time.perf_counter()
    final_training_nll = initial_nll
    for _ in range(local.max_epochs):
        model.train()
        total = 0.0
        count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=dtype)
            targets = targets.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * targets.numel()
            count += targets.numel()
        final_training_nll = total / count
    final_nll, final_accuracy, fitted_log_probs = prediction_snapshot()
    predictive_kl = torch.sum(
        reference_log_probs.exp() * (reference_log_probs - fitted_log_probs), dim=1
    ).mean()
    return {
        "initial_nll": initial_nll,
        "initial_accuracy": initial_accuracy,
        "final_training_nll": final_training_nll,
        "final_nll": final_nll,
        "final_accuracy": final_accuracy,
        "nll_reduction": initial_nll - final_nll,
        "predictive_kl_from_reference": float(predictive_kl),
        "epochs": local.max_epochs,
        "wall_time_seconds": time.perf_counter() - started,
    }


def _collect_scores(
    model: nn.Module,
    dataset: Dataset,
    layout: ParameterLayout,
    config: Phase6OracleConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    loader = DataLoader(
        dataset,
        batch_size=config.fisher.chunk_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        persistent_workers=config.runtime.num_workers > 0,
    )
    pieces = []
    model.eval()
    for inputs, targets in loader:
        derivatives = per_sample_derivatives(
            model,
            inputs.to(device=device, dtype=dtype),
            targets.to(device=device),
            mnist_nll,
            layout,
            strategy="vmap",
        )
        pieces.append(derivatives.gradients.detach().to(dtype=torch.float64))
    return torch.cat(pieces, dim=0)


def _source_schedules(source, condition: str) -> dict[str, tuple[float, ...]]:
    return {
        schedule: tuple(
            float(row["angle_degrees"])
            for row in source.trajectory_metrics[schedule][condition]
        )
        for schedule in source.config.schedule_kinds
    }


def _reference_contract(
    config: Phase6OracleConfig,
    source,
    angles: Sequence[float],
    fit_indices: Sequence[int],
    fisher_indices: Sequence[int],
    validation_indices: Sequence[int],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "source_run_id": source.config.run_id,
        "source_partition_hash": source.partitions.content_hash,
        "angles_degrees": list(angles),
        "reference": config.to_mapping()["reference"],
        "fisher": config.to_mapping()["fisher"],
        "fit_indices_hash": _canonical_hash(list(fit_indices)),
        "fisher_indices_hash": _canonical_hash(list(fisher_indices)),
        "validation_indices_hash": _canonical_hash(list(validation_indices)),
        "parameter_count": 512,
        "continuation": "ascending canonical angle; best validation state warm-starts next angle",
        "local_pairing": (
            "replicate sample identities and loader seeds are shared across angles"
        ),
    }
    value["content_hash"] = _canonical_hash(value)
    return value


def _load_reused_references(
    repo_root: Path,
    configured_path: str,
    expected_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    path = repo_root / configured_path
    if not (path / "COMPLETED").is_file():
        raise ValueError("configured reference run is incomplete")
    contract = _read_json(path / "reference_contract.json")
    if contract != expected_contract:
        raise ValueError("configured reference run has a different reference contract")
    states = torch.load(path / "reference_states.pt", map_location="cpu", weights_only=False)
    fishers = torch.load(path / "reference_fishers.pt", map_location="cpu", weights_only=False)
    metrics = _read_json(path / "reference_metrics.json")
    return states, fishers, metrics


def _reference_estimates(
    config: Phase6OracleConfig,
    source,
    train_dataset: Dataset,
    test_dataset: Dataset,
    angles: Sequence[float],
    fit_indices: Sequence[int],
    fisher_indices: Sequence[int],
    validation_indices: Sequence[int],
    contract: dict[str, Any],
    session,
    *,
    device: torch.device,
    dtype: torch.dtype,
    deadline: float,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if config.reference_run_path is not None:
        return _load_reused_references(
            Path(__file__).parents[2], config.reference_run_path, contract
        )

    checkpoint_path = session.working_path / "reference_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        states = checkpoint["states"]
        parameters = checkpoint["parameters"]
        fishers = checkpoint["fishers"]
        metrics = checkpoint["metrics"]
        completed = set(checkpoint["completed"])
        previous_state = checkpoint["previous_state"]
    else:
        stored = torch.load(source.path / "model_states.pt", map_location="cpu", weights_only=False)
        initial = stored["linear"][config.condition]["initial"]
        other = stored["sigmoid"][config.condition]["initial"]
        if state_dict_hash(initial) != state_dict_hash(other):
            raise ValueError("source schedules do not share an initializer")
        states = {}
        parameters = {}
        fishers = {}
        metrics = []
        completed = set()
        previous_state = initial

    model_seed = derive_component_seed(config.replica_seed, "plan6_reference_fit")
    model, layout = build_canonical_model(model_seed, device=device, dtype=dtype)
    if layout.total_numel != 512:
        raise ValueError("Plan 6 requires the canonical 512-parameter model")
    progress = tqdm(angles, desc=f"phase6 {config.mode} references", unit="angle")
    for position, angle in enumerate(progress):
        if time.perf_counter() >= deadline:
            raise TimeoutError("Phase 6 wall-time cap reached during reference fits")
        key = angle_key(angle)
        if key in completed:
            previous_state = states[key]
            continue
        model.load_state_dict(previous_state)
        fit_pool = _materialize_rotated(
            train_dataset, fit_indices, angle=angle, config=config
        )
        validation = _materialize_rotated(
            test_dataset, validation_indices, angle=angle, config=config
        )
        fit = _fit_reference(
            model,
            fit_pool,
            validation,
            config,
            seed=derive_component_seed(
                config.replica_seed, f"plan6_reference_loader:{key}"
            ),
            max_epochs=(
                config.reference.initial_max_epochs
                if position == 0
                else config.reference.max_epochs
            ),
            device=device,
            dtype=dtype,
        )
        state = _state_dict_cpu(model)
        parameter = layout.flatten_module(model, detach=True).cpu().to(torch.float64)
        fisher_dataset = _materialize_rotated(
            train_dataset, fisher_indices, angle=angle, config=config
        )
        scores = _collect_scores(
            model, fisher_dataset, layout, config, device=device, dtype=dtype
        )
        rank_values = {}
        for rank in config.fisher.ranks:
            approximation = approximate_score_fisher(
                scores,
                rank=rank,
                seed=derive_component_seed(
                    config.replica_seed, f"plan6_fisher_lanczos:{key}:rank={rank}"
                ),
            )
            rank_values[str(rank)] = approximation.artifact_mapping()
        states[key] = state
        parameters[key] = parameter
        fishers[key] = rank_values
        row = {
            "angle_degrees": angle,
            "angle_key": key,
            "parameter_hash": tensor_content_hash(parameter),
            "state_hash": state_dict_hash(state),
            "fit": fit,
            "fisher_sample_count": scores.shape[0],
            "fisher_ranks": {
                rank: rank_values[str(rank)]["lanczos"]
                for rank in config.fisher.ranks
            },
        }
        metrics.append(row)
        completed.add(key)
        previous_state = state
        session.write_torch(
            "reference_checkpoint.pt",
            {
                "states": states,
                "parameters": parameters,
                "fishers": fishers,
                "metrics": metrics,
                "completed": sorted(completed),
                "previous_state": previous_state,
                "reference_contract_hash": contract["content_hash"],
            },
        )
        progress.set_postfix_str(
            f"angle={angle:.3f}, nll={fit['best_validation_nll']:.3f}"
        )
    return {"states": states, "parameters": parameters}, fishers, metrics


def _local_estimates(
    config: Phase6OracleConfig,
    reference_states: dict[str, Any],
    train_dataset: Dataset,
    fit_pool_indices: Sequence[int],
    local_angles: Sequence[float],
    session,
    *,
    device: torch.device,
    dtype: torch.dtype,
    deadline: float,
) -> tuple[dict[str, dict[str, Tensor]], dict[str, dict[str, list[dict[str, Any]]]]]:
    checkpoint_path = session.working_path / "local_mle_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        parameters = checkpoint["parameters"]
        metrics = checkpoint["metrics"]
    else:
        parameters = {}
        metrics = {}
    model_seed = derive_component_seed(config.replica_seed, "plan6_local_mle_samples")
    model, layout = build_canonical_model(model_seed, device=device, dtype=dtype)
    progress = tqdm(local_angles, desc=f"phase6 {config.mode} local MLEs", unit="angle")
    for angle in progress:
        key = angle_key(angle)
        parameters.setdefault(key, {})
        metrics.setdefault(key, {})
        fit_pool = _materialize_rotated(
            train_dataset, fit_pool_indices, angle=angle, config=config
        )
        for sample_size in config.local_mle.sample_sizes:
            sample_key = str(sample_size)
            existing = parameters[key].get(sample_key)
            vectors = [] if existing is None else [row for row in existing]
            rows = metrics[key].setdefault(sample_key, [])
            while len(vectors) < config.local_mle.maximum_replicates:
                if time.perf_counter() >= deadline:
                    raise TimeoutError("Phase 6 wall-time cap reached during local fits")
                replicate = len(vectors)
                sample_seed = derive_component_seed(
                    config.replica_seed,
                    f"plan6_local_mle_samples:M={sample_size}:rep={replicate}",
                )
                generator = torch.Generator().manual_seed(sample_seed)
                positions = torch.randint(
                    len(fit_pool), (sample_size,), generator=generator
                )
                sample = Subset(fit_pool, positions.tolist())
                model.load_state_dict(reference_states["states"][key])
                fit = _fit_local_mle(
                    model,
                    sample,
                    config,
                    seed=derive_component_seed(
                        config.replica_seed,
                        f"plan6_local_mle_loader:M={sample_size}:rep={replicate}",
                    ),
                    device=device,
                    dtype=dtype,
                )
                vector = layout.flatten_module(model, detach=True).cpu().to(torch.float64)
                vectors.append(vector)
                rows.append(
                    {
                        "replicate": replicate,
                        "sample_size": sample_size,
                        "sample_seed": sample_seed,
                        "sample_positions_hash": tensor_content_hash(positions),
                        "paired_across_angles": True,
                        "parameter_hash": tensor_content_hash(vector),
                        "parameter_displacement_norm": float(
                            torch.linalg.vector_norm(
                                vector - reference_states["parameters"][key]
                            )
                        ),
                        **fit,
                    }
                )
                if len(vectors) in config.local_mle.replicate_checkpoints:
                    parameters[key][sample_key] = torch.stack(vectors)
                    session.write_torch(
                        "local_mle_checkpoint.pt",
                        {"parameters": parameters, "metrics": metrics},
                    )
            parameters[key][sample_key] = torch.stack(vectors)
        progress.set_postfix_str(f"angle={angle:.3f}")
    return parameters, metrics


def _oracle_rows(
    config: Phase6OracleConfig,
    source,
    schedules: dict[str, tuple[float, ...]],
    steps: dict[str, tuple[int, ...]],
    references: dict[str, Any],
    fisher_artifacts: dict[str, Any],
    local_parameters: dict[str, dict[str, Tensor]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    estimates: dict[str, Any] = {}
    convergence: dict[str, Any] = {}
    for schedule in config.schedule_kinds:
        estimates[schedule] = {}
        convergence[schedule] = {}
        source_rows = source.trajectory_metrics[schedule][config.condition]
        for step in steps[schedule]:
            row_key = str(step)
            left = canonical_angle(schedules[schedule][step])
            right = canonical_angle(schedules[schedule][step + 1])
            left_key = angle_key(left)
            right_key = angle_key(right)
            q = 1.0 / float(source_rows[step]["controller"]["effective_size"])
            estimates[schedule][row_key] = {}
            convergence[schedule][row_key] = {}
            for sample_size in config.local_mle.sample_sizes:
                sample_key = str(sample_size)
                estimates[schedule][row_key][sample_key] = {}
                convergence[schedule][row_key][sample_key] = {}
                for rank in config.fisher.ranks:
                    rank_key = str(rank)
                    fisher = representation_from_artifact(
                        fisher_artifacts[left_key][rank_key]["representation"],
                        device="cpu",
                    ).to(dtype=torch.float64)
                    left_values = local_parameters[left_key][sample_key]
                    right_values = local_parameters[right_key][sample_key]
                    checkpoints = {}
                    for count in config.local_mle.replicate_checkpoints:
                        estimate = estimate_oracle(
                            references["parameters"][left_key],
                            references["parameters"][right_key],
                            left_values[:count],
                            right_values[:count],
                            fisher,
                            reference_sample_size=config.reference.fit_sample_size,
                            local_sample_size=sample_size,
                            q=q,
                            deployed_batch_size=config.deployed_batch_size,
                        )
                        uncertainty = bootstrap_oracle_pi(
                            references["parameters"][left_key],
                            references["parameters"][right_key],
                            left_values[:count],
                            right_values[:count],
                            fisher,
                            reference_sample_size=config.reference.fit_sample_size,
                            local_sample_size=sample_size,
                            q=q,
                            deployed_batch_size=config.deployed_batch_size,
                            replicates=config.local_mle.bootstrap_replicates,
                            seed=derive_component_seed(
                                config.replica_seed,
                                f"plan6_bootstrap:{schedule}:step={step}:M={sample_size}:rank={rank}:B={count}",
                            ),
                        )
                        six_se = (
                            config.local_mle.sigma_multiple
                            * uncertainty["standard_error"]
                        )
                        checkpoints[str(count)] = {
                            **estimate.mapping(),
                            "bootstrap": uncertainty,
                            "six_standard_error_half_width": six_se,
                            "precision_pass": (
                                six_se
                                <= config.local_mle.pi_half_width_tolerance
                            ),
                        }
                    estimates[schedule][row_key][sample_key][rank_key] = checkpoints
                    final = checkpoints[str(config.local_mle.maximum_replicates)]
                    convergence[schedule][row_key][sample_key][rank_key] = {
                        "final_pi": final["pi"],
                        "six_standard_error_half_width": final[
                            "six_standard_error_half_width"
                        ],
                        "precision_pass": final["precision_pass"],
                    }
    return estimates, convergence


def run_oracle(
    config: Phase6OracleConfig,
    *,
    data_root: Path,
    output_root: Path,
    repo_root: Path,
    download: bool,
    resume: bool,
) -> Path:
    global _SOURCE_ROTATION
    source_path = repo_root / config.source_run_path
    source = load_completed_single_lap_run(source_path)
    if source.config.run_id != config.source_run_id:
        raise ValueError("source run does not match Plan 6")
    _SOURCE_ROTATION = source.config.rotation
    schedules = _source_schedules(source, config.condition)
    steps = transition_steps(
        schedules, config.transition_steps, full=config.mode == "full"
    )
    local_angles = required_angles(schedules, steps)
    reference_angles = (
        local_angles
        if config.mode == "smoke"
        else full_reference_angle_union(schedules)
    )

    device = resolve_device(config.runtime.device)
    dtype = resolve_dtype(config.runtime.dtype)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=device.type == "cuda",
    )
    session = Phase6RunStore(
        output_root,
        run_kind=f"phase6_{config.mode}_instantaneous_oracle",
        label=f"Phase 6 {config.mode} oracle",
    ).begin(config, repo_root, resume=resume)
    started = time.perf_counter()
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=download)
    if dataset_targets(train_dataset).numel() != source.partitions.train_size:
        raise ValueError("MNIST training dataset differs from source artifact")

    reference_indices = source.partitions.reference
    fit_pool_indices = tuple(reference_indices[: config.reference.fit_pool_size])
    fit_seed = derive_component_seed(config.replica_seed, "plan6_reference_fit")
    order = torch.randperm(
        len(fit_pool_indices), generator=torch.Generator().manual_seed(fit_seed)
    )
    fit_indices = tuple(
        fit_pool_indices[int(position)]
        for position in order[: config.reference.fit_sample_size]
    )
    fisher_indices = tuple(
        reference_indices[
            config.reference.fit_pool_size :
            config.reference.fit_pool_size + config.fisher.sample_size
        ]
    )
    validation_indices = tuple(
        source.partitions.evaluation[: config.reference.validation_size]
    )
    contract = _reference_contract(
        config,
        source,
        reference_angles,
        fit_indices,
        fisher_indices,
        validation_indices,
    )
    session.write_json(
        "source_contract.json",
        {
            "source_run_id": source.config.run_id,
            "source_config_hash": source.config.config_hash,
            "source_partition_hash": source.partitions.content_hash,
            "condition": config.condition,
            "deployed_batch_size": config.deployed_batch_size,
            "schedule_hashes": source.run_summary["schedule_hashes"],
            "transition_steps": {
                name: list(values) for name, values in steps.items()
            },
            "transition_alignment": "row t weights the transition t to t+1",
        },
    )
    session.write_json("reference_contract.json", contract)
    references, fishers, reference_metrics = _reference_estimates(
        config,
        source,
        train_dataset,
        test_dataset,
        reference_angles,
        fit_indices,
        fisher_indices,
        validation_indices,
        contract,
        session,
        device=device,
        dtype=dtype,
        deadline=started + config.max_wall_time_seconds,
    )
    if time.perf_counter() - started > config.max_wall_time_seconds:
        raise TimeoutError("Phase 6 wall-time cap reached after reference estimation")
    local_parameters, local_metrics = _local_estimates(
        config,
        references,
        train_dataset,
        fit_pool_indices,
        local_angles,
        session,
        device=device,
        dtype=dtype,
        deadline=started + config.max_wall_time_seconds,
    )
    estimates, convergence = _oracle_rows(
        config,
        source,
        schedules,
        steps,
        references,
        fishers,
        local_parameters,
    )
    final_checks = [
        details["precision_pass"]
        for schedule in convergence.values()
        for step in schedule.values()
        for sample in step.values()
        for details in sample.values()
    ]
    elapsed = time.perf_counter() - started
    session.write_json("reference_metrics.json", reference_metrics)
    session.write_torch("reference_states.pt", references)
    session.write_torch("reference_fishers.pt", fishers)
    session.write_json("local_mle_metrics.json", local_metrics)
    session.write_torch("local_mle_parameters.pt", local_parameters)
    session.write_json("oracle_estimates.json", estimates)
    session.write_json(
        "convergence.json",
        {
            "transition_results": convergence,
            "precision_pass_fraction": sum(final_checks) / len(final_checks),
            "all_precision_pass": all(final_checks),
            "sigma_multiple": config.local_mle.sigma_multiple,
            "pi_half_width_tolerance": config.local_mle.pi_half_width_tolerance,
        },
    )
    session.write_json(
        "run_summary.json",
        {
            "run_kind": f"phase6_{config.mode}_instantaneous_oracle",
            "config_hash": config.config_hash,
            "source_run_id": source.config.run_id,
            "reference_contract_hash": contract["content_hash"],
            "reference_angle_count": len(reference_angles),
            "local_angle_count": len(local_angles),
            "transition_count": sum(len(values) for values in steps.values()),
            "reference_fit_count": len(reference_metrics),
            "local_fit_count": sum(
                len(rows)
                for angle in local_metrics.values()
                for rows in angle.values()
            ),
            "parameter_count": 512,
            "deployed_batch_size": config.deployed_batch_size,
            "oracle_sample_sizes": list(config.local_mle.sample_sizes),
            "fisher_ranks": list(config.fisher.ranks),
            "precision_pass_fraction": sum(final_checks) / len(final_checks),
            "all_precision_pass": all(final_checks),
            "wall_time_seconds": elapsed,
            "wall_time_cap_seconds": config.max_wall_time_seconds,
        },
    )
    return session.complete(required=PHASE6_ORACLE_REQUIRED_ARTIFACTS)


def main() -> None:
    arguments = parse_arguments()
    config = load_phase6_oracle_config(arguments.config)
    path = run_oracle(
        config,
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        repo_root=Path(__file__).parents[2],
        download=arguments.download,
        resume=arguments.resume,
    )
    print(json.dumps({"path": str(path), "status": "completed"}, indent=2))


if __name__ == "__main__":
    main()
