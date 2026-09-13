"""Held-out finite excess-risk map for Plan 10 Phase 1b."""

from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from src.initialization import state_dict_hash
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import build_canonical_model, configure_torch_runtime, resolve_device
from src.representations import representation_from_artifact

from ..phase5_single_lap_artifacts import load_completed_single_lap_run
from ..phase6_artifacts import load_completed_phase6_oracle
from ..phase6_oracle import angle_key
from ..transform import rotate_mnist_batch, tensor_content_hash
from .artifacts import file_sha256, read_json, validate_completed
from .config import FeasibilityConfig
from .feasibility import _corrected_energy
from .finite_risk_config import FiniteRiskConfig
from .theory import fixed_pi_q_update, movement_energy_target


DERIVATIVE_REQUIRED = (
    "config.json",
    "source_contract.json",
    "fisher_speed_map.json",
    "finite_step_validation.json",
    "oracle_route.json",
    "summary.json",
)


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _grid(increment: float) -> tuple[float, ...]:
    count = round(30.0 / increment)
    values = tuple(round(index * increment, 12) for index in range(count + 1))
    if not math.isclose(values[-1], 30.0, abs_tol=1e-12):
        raise ValueError("finite-risk grid does not partition [0, 30]")
    return values


def _base_panel(dataset, indices: tuple[int, ...], *, batch_size: int) -> tuple[Tensor, Tensor]:
    inputs = []
    targets = []
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False)
    for batch_inputs, batch_targets in loader:
        inputs.append(batch_inputs.to(device="cpu", dtype=torch.float32))
        targets.append(batch_targets.to(device="cpu", dtype=torch.long))
    return torch.cat(inputs).contiguous(), torch.cat(targets).contiguous()


def _mean_nll(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    total = 0.0
    model.eval()
    with torch.inference_mode():
        for left in range(0, targets.numel(), batch_size):
            right = min(targets.numel(), left + batch_size)
            logits = model(inputs[left:right].to(device=device))
            total += float(
                nn.functional.cross_entropy(
                    logits,
                    targets[left:right].to(device=device),
                    reduction="sum",
                )
            )
    return total / targets.numel()


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = 0.5 * (position + end - 1)
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs must have equal nontrivial length")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_scale = sum((x - left_mean) ** 2 for x in left)
    right_scale = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_scale * right_scale)
    return numerator / denominator if denominator > 0.0 else 0.0


def _spearman(left: list[float], right: list[float]) -> float:
    return _correlation(_average_ranks(left), _average_ranks(right))


def build_finite_risk_map(
    config: FiniteRiskConfig,
    repo_root: Path,
    *,
    data_root: Path,
    download: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any], Tensor]:
    started = time.perf_counter()
    oracle_path = _resolve(repo_root, config.oracle_run_path)
    derivative_path = _resolve(repo_root, config.derivative_run_path)
    oracle = load_completed_phase6_oracle(oracle_path)
    derivative_manifest = validate_completed(derivative_path, required=DERIVATIVE_REQUIRED)
    derivative_config = FeasibilityConfig.from_mapping(read_json(derivative_path / "config.json"))
    if oracle_path.name != config.oracle_run_id or derivative_path.name != config.derivative_run_id:
        raise ValueError("finite-risk source identity differs from configuration")
    if derivative_config.oracle_run_id != oracle_path.name:
        raise ValueError("derivative and finite-risk oracles differ")
    source_path = _resolve(repo_root, oracle.config.source_run_path)
    source = load_completed_single_lap_run(source_path)
    if source.config.run_id != oracle.config.source_run_id:
        raise ValueError("Plan 6 oracle and source trajectory differ")
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=download)
    if dataset_targets(train_dataset).numel() != source.partitions.train_size:
        raise ValueError("MNIST training dataset differs from source partition")
    heldout = tuple(
        source.partitions.evaluation[
            config.heldout_offset : config.heldout_offset + config.heldout_size
        ]
    )
    if len(heldout) != config.heldout_size:
        raise ValueError("source partition does not contain the held-out panel")
    base_inputs, base_targets = _base_panel(
        test_dataset, heldout, batch_size=config.evaluation_batch_size
    )

    angles = _grid(config.grid_increment_degrees)
    references = torch.load(oracle_path / "reference_states.pt", map_location="cpu", weights_only=True)
    fishers = torch.load(oracle_path / "reference_fishers.pt", map_location="cpu", weights_only=True)
    local = torch.load(oracle_path / "local_mle_parameters.pt", map_location="cpu", weights_only=True)
    missing = [angle for angle in angles if angle_key(angle) not in references["states"]]
    if missing:
        raise ValueError(f"reference states lack finite-risk angles: {missing}")

    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=True,
    )
    device = resolve_device(config.runtime.device)
    model, layout = build_canonical_model(config.replica_seed, device=device, dtype=torch.float32)
    if layout.total_numel != 512:
        raise ValueError("finite-risk audit requires the canonical 512-parameter model")
    nll_matrix = torch.empty((len(angles), len(angles)), dtype=torch.float64)
    progress = tqdm(total=len(angles) ** 2, desc="Plan 10 finite NLL map", unit="evaluation")
    for target_index, target_angle in enumerate(angles):
        rotated = rotate_mnist_batch(base_inputs, target_angle, source.config.rotation)
        for model_index, model_angle in enumerate(angles):
            model.load_state_dict(references["states"][angle_key(model_angle)])
            nll_matrix[target_index, model_index] = _mean_nll(
                model,
                rotated,
                base_targets,
                batch_size=config.evaluation_batch_size,
                device=device,
            )
            progress.update(1)
    progress.close()

    derivative_rows = read_json(derivative_path / "fisher_speed_map.json")
    covariance_shapes = {float(row["angle_degrees"]): float(row["covariance_shape"]) for row in derivative_rows}
    pair_rows: list[dict[str, Any]] = []
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for from_index, from_angle in enumerate(angles):
        fisher = representation_from_artifact(
            fishers[angle_key(from_angle)][str(config.fisher_rank)]["representation"],
            device="cpu",
        ).to(dtype=torch.float64)
        for to_index, to_angle in enumerate(angles):
            if from_index == to_index:
                continue
            excess = 2.0 * float(
                nll_matrix[to_index, from_index] - nll_matrix[to_index, to_index]
            )
            raw, correction, quadratic = _corrected_energy(
                from_angle,
                to_angle,
                parameters=references["parameters"],
                local=local,
                fisher=fisher,
                local_sample_size=config.local_mle_sample_size,
                local_replicates=config.local_mle_replicates,
                reference_sample_size=oracle.config.reference.fit_sample_size,
            )
            row = {
                "from_index": from_index,
                "to_index": to_index,
                "from_angle_degrees": from_angle,
                "to_angle_degrees": to_angle,
                "direction": 1 if to_angle > from_angle else -1,
                "increment_degrees": abs(to_angle - from_angle),
                "target_nll_at_old_reference": float(nll_matrix[to_index, from_index]),
                "target_nll_at_target_reference": float(nll_matrix[to_index, to_index]),
                "finite_excess_nll_energy": excess,
                "fisher_energy_raw": raw,
                "fisher_noise_correction": correction,
                "fisher_energy": quadratic,
            }
            pair_rows.append(row)
            lookup[(from_index, to_index)] = row

    monotone_checks = []
    for from_index in range(len(angles)):
        for direction in (-1, 1):
            candidates = [
                lookup[(from_index, to_index)]
                for to_index in range(len(angles))
                if (from_index, to_index) in lookup
                and lookup[(from_index, to_index)]["direction"] == direction
            ]
            candidates.sort(key=lambda row: row["increment_degrees"])
            monotone_checks.extend(
                right["finite_excess_nll_energy"] + 1e-12
                >= left["finite_excess_nll_energy"]
                for left, right in zip(candidates, candidates[1:])
            )

    q = config.initial_q
    current_index = 0
    route_rows: list[dict[str, Any]] = []
    route_targets = []
    for leg, knot in enumerate(config.route_knots_degrees[1:]):
        knot_index = angles.index(knot)
        direction = 1 if knot_index > current_index else -1
        while current_index != knot_index:
            if len(route_rows) >= config.maximum_route_steps:
                break
            candidate_indices = list(range(current_index + direction, knot_index + direction, direction))
            candidate_details = []
            for candidate_index in candidate_indices:
                old_shape = covariance_shapes[angles[current_index]]
                new_shape = covariance_shapes[angles[candidate_index]]
                target = movement_energy_target(
                    config.fixed_pi,
                    q=q,
                    old_covariance_shape=old_shape,
                    new_covariance_shape=new_shape,
                    batch_size=config.batch_size,
                )
                pair = lookup[(current_index, candidate_index)]
                candidate_details.append((candidate_index, target, pair))
            selected = next(
                (item for item in candidate_details if item[2]["finite_excess_nll_energy"] >= item[1]),
                candidate_details[-1],
            )
            next_index, target, pair = selected
            energy = float(pair["finite_excess_nll_energy"])
            attained = energy >= target and target >= 0.0
            route_targets.append(target)
            route_rows.append(
                {
                    "step": len(route_rows),
                    "leg": leg,
                    "from_angle_degrees": angles[current_index],
                    "to_angle_degrees": angles[next_index],
                    "direction": direction,
                    "pace_degrees": abs(angles[next_index] - angles[current_index]),
                    "q": q,
                    "movement_target": target,
                    "finite_excess_nll_energy": energy,
                    "fisher_energy": pair["fisher_energy"],
                    "target_attained": attained,
                    "relative_target_error": abs(energy - target) / max(abs(energy), abs(target), 1e-12),
                    "candidate_count": len(candidate_details),
                    "cumulative_observations": (len(route_rows) + 1) * config.batch_size,
                }
            )
            q = fixed_pi_q_update(q, config.fixed_pi, config.batch_size)
            current_index = next_index
        if len(route_rows) >= config.maximum_route_steps:
            break

    nonnegative_fraction = statistics.fmean(
        float(row["finite_excess_nll_energy"] >= 0.0) for row in pair_rows
    )
    monotone_fraction = statistics.fmean(float(value) for value in monotone_checks)
    attainment_fraction = statistics.fmean(float(row["target_attained"]) for row in route_rows)
    median_target_error = statistics.median(row["relative_target_error"] for row in route_rows)
    finite_values = [row["finite_excess_nll_energy"] for row in pair_rows if row["fisher_energy"] > 0.0]
    fisher_values = [row["fisher_energy"] for row in pair_rows if row["fisher_energy"] > 0.0]
    rank_correlation = _spearman(finite_values, fisher_values)
    local_rows = [row for row in pair_rows if row["increment_degrees"] <= 3.0 and row["fisher_energy"] > 0.0]
    local_rank_correlation = _spearman(
        [row["finite_excess_nll_energy"] for row in local_rows],
        [row["fisher_energy"] for row in local_rows],
    )
    route_complete = current_index == angles.index(config.route_knots_degrees[-1])
    checks = {
        "finite_energies_mostly_nonnegative": nonnegative_fraction >= config.minimum_nonnegative_fraction,
        "nested_energies_mostly_monotone": monotone_fraction >= config.minimum_monotone_fraction,
        "route_targets_mostly_attained": attainment_fraction >= config.minimum_target_attainment_fraction,
        "route_target_error_acceptable": median_target_error <= config.maximum_median_target_error,
        "all_route_targets_nonnegative": all(value >= 0.0 for value in route_targets),
        "route_complete": route_complete and len(route_rows) < config.maximum_route_steps,
        "fisher_rank_correlation_positive": rank_correlation > 0.0,
    }
    summary = {
        "gate_pass": all(checks.values()),
        "gate_checks": checks,
        "angle_count": len(angles),
        "directed_pair_count": len(pair_rows),
        "heldout_sample_count": config.heldout_size,
        "nonnegative_energy_fraction": nonnegative_fraction,
        "nested_monotone_fraction": monotone_fraction,
        "fisher_energy_spearman": rank_correlation,
        "local_fisher_energy_spearman": local_rank_correlation,
        "route_steps": len(route_rows),
        "route_observations": len(route_rows) * config.batch_size,
        "route_target_attainment_fraction": attainment_fraction,
        "median_route_target_relative_error": median_target_error,
        "pace_minimum_degrees": min(row["pace_degrees"] for row in route_rows),
        "pace_median_degrees": statistics.median(row["pace_degrees"] for row in route_rows),
        "pace_maximum_degrees": max(row["pace_degrees"] for row in route_rows),
        "wall_time_seconds": time.perf_counter() - started,
    }
    panel_rows = [
        {
            "angle_degrees": angle,
            "own_reference_nll": float(nll_matrix[index, index]),
            "minimum_model_nll": float(nll_matrix[index].min()),
            "own_reference_is_best": int(nll_matrix[index].argmin()) == index,
        }
        for index, angle in enumerate(angles)
    ]
    source_contract = {
        "oracle_run_id": oracle_path.name,
        "derivative_run_id": derivative_path.name,
        "source_run_id": source.config.run_id,
        "source_partition_hash": source.partitions.content_hash,
        "initial_state_hash": state_dict_hash(
            torch.load(source_path / "model_states.pt", map_location="cpu", weights_only=True)["linear"][oracle.config.condition]["initial"]
        ),
        "heldout_offset": config.heldout_offset,
        "heldout_size": config.heldout_size,
        "heldout_indices_hash": tensor_content_hash(torch.tensor(heldout, dtype=torch.int64)),
        "parameter_interpolation_used": False,
        "energy_smoothing_used": False,
        "negative_energy_clamping_used": False,
        "source_file_sha256": {
            "oracle_manifest": file_sha256(oracle_path / "manifest.json"),
            "reference_states": file_sha256(oracle_path / "reference_states.pt"),
            "derivative_manifest": file_sha256(derivative_path / "manifest.json"),
        },
        "derivative_config_hash": derivative_manifest["config_hash"],
    }
    return pair_rows, route_rows, panel_rows, summary, source_contract, nll_matrix
