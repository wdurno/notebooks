"""Multi-start population-reference repair for Plan 10 Phase 1c."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.initialization import state_dict_hash
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import build_canonical_model, configure_torch_runtime, resolve_device
from src.seeding import derive_component_seed

from ..phase5_single_lap_artifacts import load_completed_single_lap_run
from ..phase6_artifacts import load_completed_phase6_oracle
from ..phase6_oracle import angle_key
from ..transform import rotate_mnist_batch, tensor_content_hash
from .artifacts import Plan10Session, file_sha256, validate_completed
from .finite_risk import _base_panel, _grid, _mean_nll
from .reference_config import ReferenceRepairConfig


FINITE_RISK_REQUIRED = (
    "config.json",
    "source_contract.json",
    "finite_risk_pairs.json",
    "oracle_route.json",
    "reference_panel.json",
    "nll_matrix.pt",
    "summary.json",
)


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _state_cpu(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }


def _evaluate_matrix(
    model: nn.Module,
    states: list[dict[str, Tensor]],
    angles: tuple[float, ...],
    base_inputs: Tensor,
    base_targets: Tensor,
    rotation,
    *,
    batch_size: int,
    device: torch.device,
    description: str,
) -> Tensor:
    matrix = torch.empty((len(angles), len(states)), dtype=torch.float64)
    progress = tqdm(total=len(angles) * len(states), desc=description, unit="evaluation")
    for target_index, angle in enumerate(angles):
        inputs = rotate_mnist_batch(base_inputs, angle, rotation)
        for state_index, state in enumerate(states):
            model.load_state_dict(state)
            matrix[target_index, state_index] = _mean_nll(
                model,
                inputs,
                base_targets,
                batch_size=batch_size,
                device=device,
            )
            progress.update(1)
    progress.close()
    return matrix


def _fit_candidate(
    model: nn.Module,
    initial_state: dict[str, Tensor],
    training_inputs: Tensor,
    training_targets: Tensor,
    selection_inputs: Tensor,
    selection_targets: Tensor,
    config: ReferenceRepairConfig,
    *,
    loader_seed: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    model.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    initial_nll = _mean_nll(
        model,
        selection_inputs,
        selection_targets,
        batch_size=config.batch_size,
        device=device,
    )
    best_nll = initial_nll
    best_epoch = 0
    best_state = _state_cpu(model)
    history = []
    loader = DataLoader(
        TensorDataset(training_inputs, training_targets),
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(loader_seed),
        num_workers=config.runtime.num_workers,
        persistent_workers=config.runtime.num_workers > 0,
    )
    started = time.perf_counter()
    for epoch in range(1, config.refinement_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * targets.numel()
            count += targets.numel()
        selection_nll = _mean_nll(
            model,
            selection_inputs,
            selection_targets,
            batch_size=config.batch_size,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "training_nll": total / count,
                "selection_nll": selection_nll,
            }
        )
        if selection_nll < best_nll - config.minimum_delta:
            best_nll = selection_nll
            best_epoch = epoch
            best_state = _state_cpu(model)
    return best_state, {
        "initial_selection_nll": initial_nll,
        "best_selection_nll": best_nll,
        "best_epoch": best_epoch,
        "epochs_executed": config.refinement_epochs,
        "wall_time_seconds": time.perf_counter() - started,
        "history": history,
    }


def _nested_monotone_fraction(pair_rows: list[dict[str, Any]], angle_count: int) -> float:
    lookup = {(row["from_index"], row["to_index"]): row for row in pair_rows}
    checks = []
    for from_index in range(angle_count):
        for direction in (-1, 1):
            candidates = [
                lookup[(from_index, to_index)]
                for to_index in range(angle_count)
                if (from_index, to_index) in lookup
                and lookup[(from_index, to_index)]["direction"] == direction
            ]
            candidates.sort(key=lambda row: row["increment_degrees"])
            checks.extend(
                right["finite_excess_nll_energy"] + 1e-12
                >= left["finite_excess_nll_energy"]
                for left, right in zip(candidates, candidates[1:])
            )
    return statistics.fmean(float(value) for value in checks)


def run_reference_repair_screen(
    config: ReferenceRepairConfig,
    repo_root: Path,
    session: Plan10Session,
    *,
    data_root: Path,
    download: bool,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    Tensor,
]:
    started = time.perf_counter()
    oracle_path = _resolve(repo_root, config.oracle_run_path)
    finite_path = _resolve(repo_root, config.finite_risk_run_path)
    oracle = load_completed_phase6_oracle(oracle_path)
    finite_manifest = validate_completed(finite_path, required=FINITE_RISK_REQUIRED)
    if oracle_path.name != config.oracle_run_id or finite_path.name != config.finite_risk_run_id:
        raise ValueError("reference-repair source identity differs")
    source_path = _resolve(repo_root, oracle.config.source_run_path)
    source = load_completed_single_lap_run(source_path)
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=download)
    if dataset_targets(train_dataset).numel() != source.partitions.train_size:
        raise ValueError("MNIST dataset differs from the source partition")

    reference_indices = source.partitions.reference
    fit_pool = tuple(reference_indices[: oracle.config.reference.fit_pool_size])
    fit_seed = derive_component_seed(oracle.config.replica_seed, "plan6_reference_fit")
    order = torch.randperm(len(fit_pool), generator=torch.Generator().manual_seed(fit_seed))
    fit_indices = tuple(
        fit_pool[int(position)]
        for position in order[: config.training_sample_size]
    )
    selection_indices = tuple(source.partitions.evaluation[: config.selection_size])
    heldout_indices = tuple(
        source.partitions.evaluation[
            config.selection_size : config.selection_size + config.heldout_size
        ]
    )
    training_base, training_targets = _base_panel(
        train_dataset, fit_indices, batch_size=config.batch_size
    )
    selection_base, selection_targets = _base_panel(
        test_dataset, selection_indices, batch_size=config.batch_size
    )
    heldout_base, heldout_targets = _base_panel(
        test_dataset, heldout_indices, batch_size=config.batch_size
    )

    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=True,
    )
    device = resolve_device(config.runtime.device)
    model, layout = build_canonical_model(config.replica_seed, device=device, dtype=torch.float32)
    angles = _grid(config.grid_increment_degrees)
    original = torch.load(oracle_path / "reference_states.pt", map_location="cpu", weights_only=True)
    original_states = [original["states"][angle_key(angle)] for angle in angles]
    original_hashes = [state_dict_hash(state) for state in original_states]
    selection_matrix = _evaluate_matrix(
        model,
        original_states,
        angles,
        selection_base,
        selection_targets,
        source.config.rotation,
        batch_size=config.batch_size,
        device=device,
        description="Plan 10 start screening",
    )

    checkpoint_path = session.working_path / "reference_screen_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        selected_states = checkpoint["states"]
        selected_parameters = checkpoint["parameters"]
        fit_rows = checkpoint["fit_rows"]
    else:
        selected_states = {}
        selected_parameters = {}
        fit_rows = []
    progress = tqdm(angles, desc="Plan 10 reference repair", unit="angle")
    for target_index, target_angle in enumerate(angles):
        key = angle_key(target_angle)
        if key in selected_states:
            progress.update(1)
            continue
        order_indices = torch.argsort(selection_matrix[target_index]).tolist()
        starts = []
        seen_hashes = set()
        for index in order_indices:
            state_hash = original_hashes[index]
            if state_hash in seen_hashes:
                continue
            starts.append(index)
            seen_hashes.add(state_hash)
            if len(starts) == config.candidate_start_count:
                break
        if len(starts) != config.candidate_start_count:
            raise ValueError("not enough distinct retained starts")
        training_inputs = rotate_mnist_batch(
            training_base, target_angle, source.config.rotation
        )
        selection_inputs = rotate_mnist_batch(
            selection_base, target_angle, source.config.rotation
        )
        candidates = []
        loader_seed = derive_component_seed(
            config.replica_seed, f"phase1c_loader:{key}"
        )
        for candidate_position, source_index in enumerate(starts):
            state, metrics = _fit_candidate(
                model,
                original_states[source_index],
                training_inputs,
                training_targets,
                selection_inputs,
                selection_targets,
                config,
                loader_seed=loader_seed,
                device=device,
            )
            candidates.append((state, metrics, source_index))
            fit_rows.append(
                {
                    "target_angle_degrees": target_angle,
                    "candidate_position": candidate_position,
                    "source_angle_degrees": angles[source_index],
                    "source_state_hash": original_hashes[source_index],
                    "loader_seed": loader_seed,
                    **metrics,
                }
            )
        selected_position = min(
            range(len(candidates)),
            key=lambda index: candidates[index][1]["best_selection_nll"],
        )
        selected_state, selected_metrics, source_index = candidates[selected_position]
        model.load_state_dict(selected_state)
        selected_states[key] = selected_state
        selected_parameters[key] = layout.flatten_module(model, detach=True).cpu().to(torch.float64)
        for row in fit_rows[-config.candidate_start_count :]:
            row["selected"] = row["candidate_position"] == selected_position
            row["selected_source_angle_degrees"] = angles[source_index]
            row["selected_best_epoch"] = selected_metrics["best_epoch"]
        session.write_torch(
            "reference_screen_checkpoint.pt",
            {
                "states": selected_states,
                "parameters": selected_parameters,
                "fit_rows": fit_rows,
            },
        )
        progress.set_postfix_str(
            f"angle={target_angle:.2f}, nll={selected_metrics['best_selection_nll']:.3f}"
        )
        progress.update(1)
    progress.close()

    repaired_state_list = [selected_states[angle_key(angle)] for angle in angles]
    heldout_matrix = _evaluate_matrix(
        model,
        repaired_state_list,
        angles,
        heldout_base,
        heldout_targets,
        source.config.rotation,
        batch_size=config.batch_size,
        device=device,
        description="Plan 10 repaired heldout audit",
    )
    original_matrix = torch.load(finite_path / "nll_matrix.pt", map_location="cpu", weights_only=True)
    panel_rows = []
    regrets = []
    for index, angle in enumerate(angles):
        own = float(heldout_matrix[index, index])
        best = float(heldout_matrix[index].min())
        regret = own - best
        regrets.append(regret)
        panel_rows.append(
            {
                "angle_degrees": angle,
                "repaired_own_nll": own,
                "best_repaired_nll": best,
                "repaired_own_is_best": int(heldout_matrix[index].argmin()) == index,
                "heldout_regret": regret,
                "original_own_nll": float(original_matrix[index, index]),
                "own_nll_improvement": float(original_matrix[index, index]) - own,
            }
        )
    pair_rows = []
    for from_index, from_angle in enumerate(angles):
        for to_index, to_angle in enumerate(angles):
            if from_index == to_index:
                continue
            pair_rows.append(
                {
                    "from_index": from_index,
                    "to_index": to_index,
                    "from_angle_degrees": from_angle,
                    "to_angle_degrees": to_angle,
                    "direction": 1 if to_angle > from_angle else -1,
                    "increment_degrees": abs(to_angle - from_angle),
                    "finite_excess_nll_energy": 2.0
                    * float(heldout_matrix[to_index, from_index] - heldout_matrix[to_index, to_index]),
                }
            )
    nonnegative_fraction = statistics.fmean(
        float(row["finite_excess_nll_energy"] >= 0.0) for row in pair_rows
    )
    local = [
        row
        for row in pair_rows
        if row["increment_degrees"] <= config.local_increment_limit_degrees
    ]
    local_nonnegative_fraction = statistics.fmean(
        float(row["finite_excess_nll_energy"] >= 0.0) for row in local
    )
    own_best_fraction = statistics.fmean(
        float(row["repaired_own_is_best"]) for row in panel_rows
    )
    median_regret = statistics.median(regrets)
    monotone_fraction = _nested_monotone_fraction(pair_rows, len(angles))
    checks = {
        "own_reference_best_fraction": own_best_fraction >= config.minimum_own_best_fraction,
        "median_heldout_regret": median_regret <= config.maximum_median_regret,
        "finite_energies_mostly_nonnegative": nonnegative_fraction >= config.minimum_nonnegative_fraction,
        "local_energies_mostly_nonnegative": local_nonnegative_fraction >= config.minimum_local_nonnegative_fraction,
        "nested_energies_mostly_monotone": monotone_fraction >= config.minimum_monotone_fraction,
    }
    summary = {
        "gate_pass": all(checks.values()),
        "gate_checks": checks,
        "angle_count": len(angles),
        "candidate_fits": len(fit_rows),
        "epochs_per_fit": config.refinement_epochs,
        "own_reference_best_fraction": own_best_fraction,
        "median_heldout_regret": median_regret,
        "maximum_heldout_regret": max(regrets),
        "nonnegative_energy_fraction": nonnegative_fraction,
        "local_nonnegative_energy_fraction": local_nonnegative_fraction,
        "nested_monotone_fraction": monotone_fraction,
        "median_own_nll_improvement": statistics.median(
            row["own_nll_improvement"] for row in panel_rows
        ),
        "wall_time_seconds": time.perf_counter() - started,
    }
    source_contract = {
        "oracle_run_id": oracle_path.name,
        "finite_risk_run_id": finite_path.name,
        "finite_risk_config_hash": finite_manifest["config_hash"],
        "source_run_id": source.config.run_id,
        "source_partition_hash": source.partitions.content_hash,
        "fit_indices_hash": tensor_content_hash(torch.tensor(fit_indices, dtype=torch.int64)),
        "selection_indices_hash": tensor_content_hash(torch.tensor(selection_indices, dtype=torch.int64)),
        "heldout_indices_hash": tensor_content_hash(torch.tensor(heldout_indices, dtype=torch.int64)),
        "selection_heldout_overlap": bool(set(selection_indices) & set(heldout_indices)),
        "source_file_sha256": {
            "oracle_manifest": file_sha256(oracle_path / "manifest.json"),
            "reference_states": file_sha256(oracle_path / "reference_states.pt"),
            "finite_risk_manifest": file_sha256(finite_path / "manifest.json"),
        },
    }
    references = {"states": selected_states, "parameters": selected_parameters}
    return references, fit_rows, panel_rows, pair_rows, summary, source_contract, heldout_matrix
