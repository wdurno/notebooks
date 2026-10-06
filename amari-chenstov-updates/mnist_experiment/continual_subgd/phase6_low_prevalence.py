"""Plan 13 Phase 6: prospective low-prevalence digit-9 few-shot study."""

from __future__ import annotations

import argparse
import dataclasses
import math
import resource
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from src.config import DataConfig, InitializationConfig, ScheduleConfig
from src.derivatives import per_sample_derivatives
from src.hybrid import blend_archive_fisher
from src.initialization import fit_p0_initialization
from src.mnist_data import (
    MixtureStreamPlan,
    dataset_targets,
    generate_mixture_stream,
    load_mnist_datasets,
    partition_mnist,
)
from src.mnist_model import mnist_nll
from src.representations import LowRankDiagonalFisher, representation_from_artifact
from src.schedules import resolve_schedule
from src.seeding import derive_component_seed

from mnist_experiment.rotated_mnist.artifacts import _read_json
from mnist_experiment.rotated_mnist.plan12.gauge import (
    build_gauge_fixed_model,
    helmert_contrast,
)
from mnist_experiment.rotated_mnist.run import _state_dict_cpu
from mnist_experiment.rotated_mnist.run_phase8 import _auc, _finite_tree
from mnist_experiment.rotated_mnist.transform import tensor_content_hash

from .artifacts import UnitStore, file_hash, freeze_json
from .conditions import Condition, condition_from_mapping
from .config import Plan13Study, canonical_hash
from .controller import InnovationControllerConfig, InnovationControllerState
from .coordinate_lbfgs import (
    CoordinateFactor,
    geometry_factor,
    identity_factor,
    selection_factor,
)
from .environments.digit9_mixture import MIXTURE_PI
from .geometry import AdaptationGeometry, random_geometry
from .phase5r import _optimize, _optimizer_contract, _paired_effect, _training_geometry
from .refresh_notebook import DEFAULT_NOTEBOOK, refresh
from .rotation import runtime
from .trajectory import _fresh_fisher, _head_mask


DEFAULT_CONFIG = Path("mnist_experiment/continual_subgd/configs/default.json")
DEFAULT_ROOT = Path("cache/mnist_experiment/continual_subgd/default")
DEFAULT_DATA_ROOT = Path("cache/mnist_experiment/datasets")
PHASE = "phase6_low_prevalence"
SMOKE_PHASE = "phase6_low_prevalence_smoke"
CUDA_SMOKE_PHASE = "phase6_low_prevalence_cuda_smoke"
SCHEMA_VERSION = "plan13-phase6-low-prevalence-v1"
PHASE6A = "phase6a_ten_positive"
PHASE6A_SMOKE = "phase6a_ten_positive_smoke"
PHASE6A_CUDA_SMOKE = "phase6a_ten_positive_cuda_smoke"
PHASE6A_SCHEMA_VERSION = "plan13-phase6a-ten-positive-v1"
PHASE6B = "phase6b_hundred_positive"
PHASE6B_SMOKE = "phase6b_hundred_positive_smoke"
PHASE6B_CUDA_SMOKE = "phase6b_hundred_positive_cuda_smoke"
PHASE6B_SCHEMA_VERSION = "plan13-phase6b-hundred-positive-v1"
PRODUCTION_REPLICAS = 64
BURN_IN_STEPS = 1
RANK_CAP = 16
REFERENCE_PREVALENCE = 0.1
PR_THRESHOLDS = 201
ASSET_REQUIRED = ("assets.pt", "summary.json", "checks.json")
BURN_REQUIRED = ("burn_in.pt", "metrics.json", "summary.json", "checks.json")
TRAJECTORY_REQUIRED = (
    "trajectory.pt",
    "metrics.json",
    "pr_curves.json",
    "summary.json",
    "checks.json",
    "final_state.pt",
)
ANALYSIS_REQUIRED = ("summary.json", "checks.json")


def phase6_conditions() -> tuple[Condition, ...]:
    controller = InnovationControllerConfig(
        alpha_scale=8.0,
        beta_scale=4.0,
        innovation_half_life=8.0,
        beta_min_half_life=64.0,
        beta_max_half_life=8.0,
    )
    return (
        Condition("no_update", "no_update"),
        Condition("full_space", "full_space"),
        Condition("digit9_bias_only", "bias_only"),
        Condition("head_only", "head_only"),
        Condition("random_rank_one", "random_rank_matched"),
        Condition("static_tiny_burn_subgd", "static_subgd"),
        Condition(
            "adaptive_floor_0.1",
            "adaptive_subgd",
            epsilon=0.1,
            controller=controller,
        ),
    )


def phase6a_conditions() -> tuple[Condition, ...]:
    retained = {
        "no_update",
        "full_space",
        "digit9_bias_only",
        "head_only",
        "adaptive_floor_0.1",
    }
    return tuple(
        condition for condition in phase6_conditions() if condition.name in retained
    )


def phase6b_conditions() -> tuple[Condition, ...]:
    return phase6a_conditions()


def _is_phase6a(phase: str) -> bool:
    return phase.startswith("phase6a_ten_positive")


def _is_phase6b(phase: str) -> bool:
    return phase.startswith("phase6b_hundred_positive")


def _is_scaling_phase(phase: str) -> bool:
    return _is_phase6a(phase) or _is_phase6b(phase)


def _schema_version(phase: str) -> str:
    if _is_phase6b(phase):
        return PHASE6B_SCHEMA_VERSION
    if _is_phase6a(phase):
        return PHASE6A_SCHEMA_VERSION
    return SCHEMA_VERSION


def _conditions(phase: str) -> tuple[Condition, ...]:
    return phase6b_conditions() if _is_phase6b(phase) else (
        phase6a_conditions() if _is_phase6a(phase) else phase6_conditions()
    )


def _samples_per_step(*, phase: str, smoke: bool) -> int:
    if _is_phase6b(phase):
        return 182
    if _is_phase6a(phase):
        return 18
    return 2 if smoke else 8


def _expected_treatment_nines(*, phase: str, smoke: bool) -> float:
    if _is_phase6b(phase):
        return 100.1
    if _is_phase6a(phase):
        return 9.9
    return 1.1 if smoke else 4.4


def low_prevalence_data_config(*, smoke: bool, phase: str = PHASE) -> DataConfig:
    return DataConfig(
        num_p_steps=11,
        samples_per_step=_samples_per_step(phase=phase, smoke=smoke),
        non_nine_sampling="empirical",
        initialization_size=512 if smoke else 30_000,
        online_pool_size=512 if smoke else 12_000,
        reference_pool_size=512 if smoke else 12_000,
        evaluation_size=512 if smoke else 10_000,
        schedule=ScheduleConfig(
            kind="linear",
            p_start=0.0,
            p_end=0.1,
            center_fraction=None,
            steepness=None,
        ),
    )


def expected_schedule(*, phase: str = PHASE) -> tuple[float, ...]:
    # Freeze the exact binary floats emitted by the versioned schedule resolver.
    return resolve_schedule(
        low_prevalence_data_config(smoke=False, phase=phase)
    ).p_values


def fixed_prevalence_precision(
    recall: float,
    specificity: float,
    *,
    prevalence: float = REFERENCE_PREVALENCE,
) -> tuple[float, bool]:
    if not 0 <= recall <= 1 or not 0 <= specificity <= 1:
        raise ValueError("recall and specificity must lie in [0, 1]")
    if not 0 <= prevalence <= 1:
        raise ValueError("prevalence must lie in [0, 1]")
    true_positive_mass = prevalence * recall
    false_positive_mass = (1 - prevalence) * (1 - specificity)
    denominator = true_positive_mass + false_positive_mass
    if denominator == 0:
        return 0.0, True
    return true_positive_mass / denominator, False


def standardized_precision_recall_curve(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    prevalence: float = REFERENCE_PREVALENCE,
    threshold_count: int = PR_THRESHOLDS,
) -> dict[str, Any]:
    if scores.ndim != 1 or targets.ndim != 1 or scores.shape != targets.shape:
        raise ValueError("scores and targets must be matching vectors")
    if threshold_count < 2 or not torch.isfinite(scores).all():
        raise ValueError("precision-recall inputs are invalid")
    positive = targets == 9
    if not bool(positive.any()) or not bool((~positive).any()):
        raise ValueError("precision-recall evaluation requires both target strata")
    scores64 = scores.to(dtype=torch.float64)
    thresholds = torch.linspace(
        1.0,
        0.0,
        threshold_count,
        dtype=torch.float64,
        device=scores.device,
    )
    predicted = scores64.unsqueeze(0) >= thresholds.unsqueeze(1)
    recall = predicted[:, positive].to(torch.float64).mean(dim=1)
    false_positive_rate = predicted[:, ~positive].to(torch.float64).mean(dim=1)
    true_positive_mass = prevalence * recall
    false_positive_mass = (1 - prevalence) * false_positive_rate
    denominator = true_positive_mass + false_positive_mass
    precision = torch.where(
        denominator > 0,
        true_positive_mass / denominator,
        torch.zeros_like(denominator),
    )

    order = torch.argsort(scores64, descending=True, stable=True)
    ordered_positive = positive[order]
    positive_weight = prevalence / int(positive.sum())
    negative_weight = (1 - prevalence) / int((~positive).sum())
    weights = torch.where(
        ordered_positive,
        torch.full_like(scores64, positive_weight)[order],
        torch.full_like(scores64, negative_weight)[order],
    )
    cumulative_true = torch.cumsum(weights * ordered_positive, dim=0)
    cumulative_total = torch.cumsum(weights, dim=0)
    ranked_precision = cumulative_true / cumulative_total
    average_precision = float(ranked_precision[ordered_positive].mean())
    return {
        "reference_prevalence": prevalence,
        "thresholds": [float(value) for value in thresholds.cpu()],
        "precision": [float(value) for value in precision.cpu()],
        "recall": [float(value) for value in recall.cpu()],
        "false_positive_rate": [float(value) for value in false_positive_rate.cpu()],
        "average_precision": average_precision,
    }


@torch.no_grad()
def evaluate_low_prevalence(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    p: float,
    *,
    include_pr_curve: bool = False,
    chunk_size: int = 1024,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    losses = []
    correct = []
    predictions = []
    nine_scores = []
    for start in range(0, targets.numel(), chunk_size):
        batch_inputs = inputs[start : start + chunk_size]
        batch_targets = targets[start : start + chunk_size]
        logits = model(batch_inputs)
        losses.append(
            torch.nn.functional.cross_entropy(logits, batch_targets, reduction="none")
        )
        predicted = logits.argmax(dim=1)
        predictions.append(predicted)
        correct.append((predicted == batch_targets).to(logits.dtype))
        nine_scores.append(torch.softmax(logits, dim=1)[:, 9])
    loss = torch.cat(losses)
    accuracy = torch.cat(correct)
    predicted = torch.cat(predictions)
    score = torch.cat(nine_scores)
    nine = targets == 9
    if not bool(nine.any()) or not bool((~nine).any()):
        raise RuntimeError("Phase 6 evaluation panel lacks a target stratum")
    recall = float((predicted[nine] == 9).to(torch.float64).mean())
    specificity = float((predicted[~nine] != 9).to(torch.float64).mean())
    precision_ref, no_positive_ref = fixed_prevalence_precision(recall, specificity)
    precision_current, no_positive_current = fixed_prevalence_precision(
        recall,
        specificity,
        prevalence=p,
    )
    non_nine_nll = float(loss[~nine].mean())
    nine_nll = float(loss[nine].mean())
    non_nine_accuracy = float(accuracy[~nine].mean())
    nine_accuracy = float(accuracy[nine].mean())
    score64 = score.to(torch.float64)
    brier = REFERENCE_PREVALENCE * float((score64[nine] - 1).square().mean()) + (
        1 - REFERENCE_PREVALENCE
    ) * float(score64[~nine].square().mean())
    sample_weights = torch.where(
        nine,
        torch.full_like(score64, REFERENCE_PREVALENCE / int(nine.sum())),
        torch.full_like(score64, (1 - REFERENCE_PREVALENCE) / int((~nine).sum())),
    )
    ece = 0.0
    boundaries = torch.linspace(0, 1, 11, device=score.device, dtype=score64.dtype)
    binary_target = nine.to(torch.float64)
    for bin_index in range(10):
        selected = (score64 >= boundaries[bin_index]) & (
            score64 < boundaries[bin_index + 1]
            if bin_index < 9
            else score64 <= boundaries[bin_index + 1]
        )
        weight = sample_weights[selected].sum()
        if float(weight) == 0:
            continue
        confidence = (sample_weights[selected] * score64[selected]).sum() / weight
        observed = (sample_weights[selected] * binary_target[selected]).sum() / weight
        ece += float(weight * torch.abs(confidence - observed))
    metrics = {
        "current_nll": (1 - p) * non_nine_nll + p * nine_nll,
        "current_accuracy": (1 - p) * non_nine_accuracy + p * nine_accuracy,
        "p0_nll": non_nine_nll,
        "p0_accuracy": non_nine_accuracy,
        "p1_nll": nine_nll,
        "p1_accuracy": nine_accuracy,
        "worst_panel_nll": max(non_nine_nll, nine_nll),
        "nine_ovr_recall": recall,
        "nine_ovr_specificity": specificity,
        "nine_ovr_false_positive_rate": 1 - specificity,
        "nine_ovr_accuracy": p * recall + (1 - p) * specificity,
        "nine_ovr_balanced_accuracy": 0.5 * (recall + specificity),
        "precision_ref_0.1": precision_ref,
        "precision_current_p": precision_current,
        "no_predicted_positive_ref_0.1": no_positive_ref,
        "no_predicted_positive_current_p": no_positive_current,
        "false_positives_per_1000": 1000 * (1 - specificity),
        "nine_brier_ref_0.1": brier,
        "nine_ece_ref_0.1": ece,
    }
    curve = (
        standardized_precision_recall_curve(score, targets)
        if include_pr_curve
        else None
    )
    return metrics, curve


def digit9_bias_factor(
    layout: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> CoordinateFactor:
    specs = {spec.name: spec for spec in layout.specs}
    bias = specs.get("classifier.bias")
    if bias is None or bias.numel != 9:
        raise ValueError("unexpected gauge-fixed classifier bias layout")
    raw = torch.full((10,), -1 / 9, device=device, dtype=dtype)
    raw[9] = 1.0
    chart = helmert_contrast(device=device, dtype=dtype).mT @ raw
    direction = torch.zeros(layout.total_numel, device=device, dtype=dtype)
    direction[bias.start : bias.stop] = chart
    scale = torch.linalg.vector_norm(direction)
    return CoordinateFactor(
        "low_rank",
        layout.total_numel,
        1,
        basis=(direction / scale).unsqueeze(1),
        parallel_scales=scale.reshape(1),
    )


@torch.no_grad()
def digit9_bias_functional_error(
    model: torch.nn.Module,
    layout: Any,
    inputs: torch.Tensor,
    *,
    delta: float = 0.125,
) -> float:
    """Measure the error in the declared raw-logit bias contrast."""

    parameters = layout.flatten_module(model, detach=True)
    factor = digit9_bias_factor(
        layout,
        device=parameters.device,
        dtype=parameters.dtype,
    )
    baseline = model(inputs)
    coordinate = parameters.new_tensor([delta])
    try:
        layout.copy_vector_to_module(model, parameters + factor.apply(coordinate))
        shifted = model(inputs)
    finally:
        layout.copy_vector_to_module(model, parameters)
    expected = parameters.new_full((10,), -delta / 9)
    expected[9] = delta
    return float(torch.max(torch.abs((shifted - baseline) - expected)))


def _initial_geometry(displacement: torch.Tensor) -> AdaptationGeometry | None:
    norm = torch.linalg.vector_norm(displacement)
    threshold = 64 * torch.finfo(displacement.dtype).eps * max(float(norm), 1.0)
    if float(norm) <= threshold:
        return None
    return AdaptationGeometry(
        (displacement / norm).unsqueeze(1),
        norm.square().reshape(1),
    )


def _grow_geometry(
    geometry: AdaptationGeometry | None,
    observation: torch.Tensor,
    beta: float,
) -> AdaptationGeometry | None:
    if geometry is None:
        norm = torch.linalg.vector_norm(observation)
        threshold = 64 * torch.finfo(observation.dtype).eps * max(float(norm), 1.0)
        if float(norm) <= threshold:
            return None
        return AdaptationGeometry(
            (observation / norm).unsqueeze(1),
            (float(beta) * norm.square()).reshape(1),
        )
    return geometry.update_growing(observation, beta, rank_cap=RANK_CAP)


def _projector_distance(
    left: AdaptationGeometry | None,
    right: AdaptationGeometry | None,
) -> float:
    if left is None and right is None:
        return 0.0
    template = left if left is not None else right
    assert template is not None
    zero = torch.zeros(
        template.parameter_count,
        template.parameter_count,
        device=template.basis.device,
        dtype=template.basis.dtype,
    )
    left_projector = zero if left is None else left.basis @ left.basis.mT
    right_projector = zero if right is None else right.basis @ right.basis.mT
    return float(torch.linalg.matrix_norm(left_projector - right_projector) / math.sqrt(2))


def _materialize(dataset: Any, indices: list[int] | tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [dataset[int(index)] for index in indices]
    return torch.stack([item[0] for item in pairs]), torch.tensor(
        [int(item[1]) for item in pairs], dtype=torch.long
    )


def _initialization_config(store: UnitStore, *, smoke: bool) -> InitializationConfig:
    source = store.study.protocol.initialization
    return InitializationConfig(
        optimizer=source.optimizer,
        learning_rate=source.learning_rate,
        weight_decay=source.weight_decay,
        batch_size=source.batch_size,
        max_epochs=1 if smoke else source.max_epochs,
        target_non_nine_accuracy=None if smoke else 0.80,
        num_workers=0 if smoke else store.study.protocol.runtime.num_workers,
    )


def _initial_fisher(
    store: UnitStore,
    model: torch.nn.Module,
    layout: Any,
    train_dataset: Any,
    reference_indices: tuple[int, ...],
    *,
    replica_seed: int,
    smoke: bool,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> tuple[LowRankDiagonalFisher, dict[str, Any]]:
    targets = dataset_targets(train_dataset)
    candidates = torch.tensor(reference_indices, dtype=torch.long)
    candidates = candidates[targets[candidates] != 9]
    sample_size = 32 if smoke else 20_000
    generator = torch.Generator().manual_seed(
        derive_component_seed(replica_seed, "initial_fisher")
    )
    selected = candidates[
        torch.randint(candidates.numel(), (sample_size,), generator=generator)
    ]
    dense = torch.zeros(
        layout.total_numel,
        layout.total_numel,
        device=device,
        dtype=matrix_dtype,
    )
    chunk_size = 16 if smoke else 512
    started = time.perf_counter()
    for start in range(0, sample_size, chunk_size):
        inputs, labels = _materialize(
            train_dataset,
            selected[start : start + chunk_size].tolist(),
        )
        gradients = per_sample_derivatives(
            model,
            inputs.to(device=device, dtype=training_dtype),
            labels.to(device=device),
            mnist_nll,
            layout,
            strategy="vmap",
        ).gradients.to(dtype=matrix_dtype)
        dense.add_(gradients.mT @ gradients)
    dense.div_(sample_size)
    dense = (dense + dense.mT) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(dense)
    values = eigenvalues[-8:].clamp_min(0)
    vectors = eigenvectors[:, -8:]
    factor = vectors * values.sqrt().unsqueeze(0)
    residual = (torch.diagonal(dense) - factor.square().sum(dim=1)).clamp_min(0)
    fisher = LowRankDiagonalFisher(factor, residual)
    return fisher, {
        "sample_size": sample_size,
        "rank": fisher.rank,
        "trace": float(fisher.diagonal_vector().sum()),
        "wall_seconds": time.perf_counter() - started,
        "sampling": "with_replacement_from_independent_p0_reference_pool",
    }


def asset_unit(
    store: UnitStore,
    index: int,
    *,
    phase: str,
    smoke: bool,
) -> dict[str, Any]:
    data = low_prevalence_data_config(smoke=smoke, phase=phase)
    schedule = resolve_schedule(data)
    return store.unit(
        phase,
        "low_prevalence_assets",
        index,
        environment="digit9_mixture",
        detail={
            "schema_version": _schema_version(phase),
            "data": data.to_mapping(),
            "p_values": list(schedule.p_values),
            "schedule_hash": schedule.content_hash,
            "initial_fisher_samples": 32 if smoke else 20_000,
            "fisher_rank": 8,
            "fresh_seed_family": phase,
        },
    )


def burn_unit(
    store: UnitStore,
    index: int,
    *,
    phase: str,
    smoke: bool,
) -> dict[str, Any]:
    source = asset_unit(store, index, phase=phase, smoke=smoke)
    return store.unit(
        phase,
        "low_prevalence_burn_in",
        index,
        environment="digit9_mixture",
        detail={
            "schema_version": _schema_version(phase),
            "burn_in_steps": BURN_IN_STEPS,
            "burn_in_p": 0.0,
            "asset_unit_hash": canonical_hash(source),
            "optimizer": _optimizer_contract(store),
        },
    )


def trajectory_unit(
    store: UnitStore,
    index: int,
    condition: Condition,
    *,
    phase: str,
    smoke: bool,
) -> dict[str, Any]:
    data = low_prevalence_data_config(smoke=smoke, phase=phase)
    return store.unit(
        phase,
        "low_prevalence_trajectory",
        index,
        environment="digit9_mixture",
        condition=condition.name,
        detail={
            "schema_version": _schema_version(phase),
            "condition": condition.mapping(),
            "burn_unit_hash": canonical_hash(
                burn_unit(store, index, phase=phase, smoke=smoke)
            ),
            "schedule": list(expected_schedule(phase=phase)),
            "samples_per_step": data.samples_per_step,
            "expected_treatment_nines": data.samples_per_step
            * sum(expected_schedule(phase=phase)[1:]),
            "reference_prevalence": REFERENCE_PREVALENCE,
            "rank_cap": RANK_CAP,
            "optimizer": _optimizer_contract(store),
        },
    )


def analysis_unit(
    store: UnitStore,
    replicas: tuple[int, ...],
    *,
    phase: str,
    smoke: bool,
) -> dict[str, Any]:
    conditions = _conditions(phase)
    return store.unit(
        phase,
        "low_prevalence_analysis",
        0,
        environment="digit9_mixture",
        detail={
            "schema_version": _schema_version(phase),
            "replicas": list(replicas),
            "conditions": [condition.mapping() for condition in conditions],
            "trajectory_unit_hashes": [
                canonical_hash(
                    trajectory_unit(
                        store,
                        index,
                        condition,
                        phase=phase,
                        smoke=smoke,
                    )
                )
                for index in replicas
                for condition in conditions
            ],
        },
    )


def ensure_assets(
    store: UnitStore,
    index: int,
    *,
    phase: str,
    smoke: bool,
    data_root: Path,
    resume: bool,
) -> Path:
    unit = asset_unit(store, index, phase=phase, smoke=smoke)
    session = store.begin(unit, ASSET_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ASSET_REQUIRED)
        assert completed is not None
        return completed
    device, training_dtype, matrix_dtype = runtime(store.study)
    data = low_prevalence_data_config(smoke=smoke, phase=phase)
    replica_seed = store.study.seed(f"{phase}:replica", index)
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=False)
    train_targets = dataset_targets(train_dataset)
    test_targets = dataset_targets(test_dataset)
    partitions = partition_mnist(
        train_targets,
        test_targets,
        data,
        replica_seed=replica_seed,
    )
    stream = generate_mixture_stream(
        train_targets,
        partitions,
        data,
        seed=derive_component_seed(replica_seed, "online_stream"),
    )
    model, layout = build_gauge_fixed_model(
        derive_component_seed(replica_seed, "model_initialization"),
        device=device,
        dtype=training_dtype,
    )
    initialization = fit_p0_initialization(
        model,
        train_dataset,
        test_dataset,
        partitions,
        _initialization_config(store, smoke=smoke),
        loader_seed=derive_component_seed(replica_seed, "initialization_loader"),
        device=device,
        dtype=training_dtype,
    )
    fisher, fisher_summary = _initial_fisher(
        store,
        model,
        layout,
        train_dataset,
        partitions.reference,
        replica_seed=replica_seed,
        smoke=smoke,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
    )
    flat_indices = [value for row in stream.observation_indices for value in row]
    stream_inputs, stream_targets = _materialize(train_dataset, flat_indices)
    stream_inputs = stream_inputs.reshape(
        len(stream.p_values),
        stream.samples_per_step,
        *stream_inputs.shape[1:],
    )
    stream_targets = stream_targets.reshape(len(stream.p_values), stream.samples_per_step)
    evaluation_inputs, evaluation_targets = _materialize(
        test_dataset,
        partitions.evaluation,
    )
    realized_nines = [int((row == 9).sum()) for row in stream_targets]
    bias_functional_error = digit9_bias_functional_error(
        model,
        layout,
        stream_inputs[0, : min(2, stream.samples_per_step)].to(
            device=device,
            dtype=training_dtype,
        ),
    )
    assets = {
        "model": _state_dict_cpu(model),
        "parameter_layout": layout.metadata(),
        "initial_fisher": fisher.artifact_mapping(),
        "partitions": partitions.to_mapping(),
        "stream_plan": stream.to_mapping(),
        "stream_inputs": stream_inputs,
        "stream_targets": stream_targets,
        "evaluation_inputs": evaluation_inputs,
        "evaluation_targets": evaluation_targets,
    }
    checks = {
        "parameter_count_is_487": layout.total_numel == 487,
        "schedule_exact": tuple(stream.p_values) == expected_schedule(phase=phase),
        "stream_labels_match": torch.equal(
            stream_targets,
            torch.tensor(stream.class_labels, dtype=torch.long),
        ),
        "p0_burn_batch_has_no_nines": realized_nines[0] == 0,
        "fixed_non_nine_conditional": stream.non_nine_sampling == "empirical",
        "digit9_bias_functional_equivalence": bias_functional_error
        <= (5e-6 if training_dtype == torch.float32 else 1e-10),
        "fresh_seed_family": phase,
        "phase5_inputs_loaded": False,
        "phase5r_inputs_loaded": False,
    }
    if not all(value for key, value in checks.items() if not key.endswith("_loaded")):
        raise RuntimeError(f"Phase 6 asset checks failed: {checks}")
    summary = {
        "schema_version": _schema_version(phase),
        "phase": phase,
        "environment": "digit9_mixture",
        "replica_index": index,
        "replica_seed": replica_seed,
        "p_values": list(stream.p_values),
        "samples_per_step": stream.samples_per_step,
        "expected_treatment_nines": stream.samples_per_step * sum(stream.p_values[1:]),
        "realized_nines_per_step": realized_nines,
        "realized_treatment_nines": sum(realized_nines[1:]),
        "stream_hash": stream.content_hash,
        "schedule_hash": stream.schedule_hash,
        "initialization": dataclasses.asdict(initialization),
        "initial_fisher": fisher_summary,
        "digit9_bias_functional_max_abs_error": bias_functional_error,
        "model_state_hash": tensor_content_hash(
            layout.flatten_module(model, detach=True).cpu()
        ),
    }
    session.write_torch("assets.pt", assets)
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    return store.finish(session, ASSET_REQUIRED)


def _load_initial_state(
    store: UnitStore,
    assets: dict[str, Any],
    index: int,
    *,
    phase: str,
) -> tuple[torch.nn.Module, Any, LowRankDiagonalFisher]:
    device, training_dtype, matrix_dtype = runtime(store.study)
    model, layout = build_gauge_fixed_model(
        store.study.seed(f"{phase}:load_model", index),
        device=device,
        dtype=training_dtype,
    )
    model.load_state_dict(assets["model"])
    layout.assert_metadata(assets["parameter_layout"])
    fisher = representation_from_artifact(assets["initial_fisher"], device=device)
    if not isinstance(fisher, LowRankDiagonalFisher) or fisher.rank != 8:
        raise RuntimeError("Phase 6 initial Fisher must be rank-eight plus diagonal")
    return model, layout, fisher.to(device=device, dtype=matrix_dtype)


def run_burn_in(
    store: UnitStore,
    index: int,
    *,
    phase: str,
    smoke: bool,
    data_root: Path,
    resume: bool,
) -> Path:
    unit = burn_unit(store, index, phase=phase, smoke=smoke)
    session = store.begin(unit, BURN_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, BURN_REQUIRED)
        assert completed is not None
        return completed
    asset_path = store.completed(
        asset_unit(store, index, phase=phase, smoke=smoke),
        ASSET_REQUIRED,
    )
    if asset_path is None:
        asset_path = ensure_assets(
            store,
            index,
            phase=phase,
            smoke=smoke,
            data_root=data_root,
            resume=True,
        )
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    plan = MixtureStreamPlan.from_mapping(assets["stream_plan"])
    if (
        tuple(plan.p_values) != expected_schedule(phase=phase)
        or float(plan.p_values[0]) != 0
    ):
        raise RuntimeError("Phase 6 burn-in received an incompatible schedule")
    device, training_dtype, matrix_dtype = runtime(store.study)
    model, layout, fisher = _load_initial_state(store, assets, index, phase=phase)
    evaluation_inputs = assets["evaluation_inputs"].to(device=device, dtype=training_dtype)
    evaluation_targets = assets["evaluation_targets"].to(device=device)
    initial_parameters = layout.flatten_module(model, detach=True).cpu()
    initial_metrics, _ = evaluate_low_prevalence(
        model,
        evaluation_inputs,
        evaluation_targets,
        0.0,
    )
    inputs = assets["stream_inputs"][0].to(device=device, dtype=training_dtype)
    targets = assets["stream_targets"][0].to(device=device)
    if bool((targets == 9).any()):
        raise RuntimeError("Phase 6 p=0 burn-in batch contains digit 9")
    started = time.perf_counter()
    fresh, _ = _fresh_fisher(
        model,
        layout,
        inputs,
        targets,
        matrix_dtype=matrix_dtype,
    )
    learner_started = time.perf_counter()
    result = _optimize(
        store,
        model,
        layout,
        inputs,
        targets,
        fisher,
        identity_factor(layout.total_numel, device=device, dtype=training_dtype),
    )
    learner_seconds = time.perf_counter() - learner_started
    update = blend_archive_fisher(
        fisher,
        fresh,
        blend_gain=MIXTURE_PI,
        rank=8,
        lanczos_seed=derive_component_seed(
            store.study.seed(f"{phase}:replica", index),
            "burn_update_lanczos:0",
        ),
    )
    fisher = update.representation
    final_parameters = layout.flatten_module(model, detach=True).cpu()
    displacement = result.displacement.to(dtype=matrix_dtype).cpu()
    if not torch.equal(final_parameters - initial_parameters, displacement.to(final_parameters)):
        raise RuntimeError("Phase 6 burn-in displacement identity failed")
    endpoint_metrics, _ = evaluate_low_prevalence(
        model,
        evaluation_inputs,
        evaluation_targets,
        0.01,
    )
    geometry = _initial_geometry(displacement)
    rows = [
        {
            "timing": "pre_burn_in",
            "step": 0,
            "p": 0.0,
            "observations_before_evaluation": 0,
            **initial_metrics,
        },
        {
            "timing": "post_burn_in_pre_treatment",
            "step": 1,
            "p": 0.01,
            "observations_before_evaluation": plan.samples_per_step,
            **endpoint_metrics,
        },
    ]
    summary = {
        "schema_version": _schema_version(phase),
        "phase": phase,
        "environment": "digit9_mixture",
        "replica_index": index,
        "burn_in_steps": BURN_IN_STEPS,
        "burn_in_p": 0.0,
        "observations_used": plan.samples_per_step,
        "effective_basis_rank": 0 if geometry is None else geometry.rank,
        "displacement_norm": float(torch.linalg.vector_norm(displacement)),
        "source_asset_sha256": file_hash(asset_path / "assets.pt"),
        "optimizer": _optimizer_contract(store),
        "learner_wall_seconds": learner_seconds,
        "total_wall_time_seconds": time.perf_counter() - started,
    }
    checks = {
        "all_finite": _finite_tree(rows) and _finite_tree(summary),
        "displacement_identity": True,
        "burn_in_only_at_p0": True,
        "effective_rank_at_most_one": geometry is None or geometry.rank <= 1,
        "source_asset_integrity_verified": True,
        "phase5_inputs_loaded": False,
        "phase5r_inputs_loaded": False,
    }
    if not all(value for key, value in checks.items() if not key.endswith("_loaded")):
        raise RuntimeError(f"Phase 6 burn-in checks failed: {checks}")
    session.write_torch(
        "burn_in.pt",
        {
            "model": _state_dict_cpu(model),
            "fisher": fisher.artifact_mapping(),
            "parameters": torch.stack((initial_parameters, final_parameters)),
            "shadow_displacement": displacement,
            "parameter_layout": layout.metadata(),
        },
    )
    session.write_json("metrics.json", rows)
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    return store.finish(session, BURN_REQUIRED)


def _factor_for_condition(
    condition: Condition,
    layout: Any,
    geometry: AdaptationGeometry | None,
    head_mask: torch.Tensor,
    *,
    alpha: float,
    device: torch.device,
    training_dtype: torch.dtype,
) -> CoordinateFactor | None:
    if condition.kind == "no_update":
        return None
    if condition.kind == "full_space":
        return identity_factor(layout.total_numel, device=device, dtype=training_dtype)
    if condition.kind == "bias_only":
        return digit9_bias_factor(layout, device=device, dtype=training_dtype)
    if condition.kind == "head_only":
        return selection_factor(head_mask)
    if geometry is None:
        if condition.kind == "adaptive_subgd":
            return identity_factor(layout.total_numel, device=device, dtype=training_dtype)
        return None
    return geometry_factor(
        _training_geometry(geometry, training_dtype),
        alpha=alpha,
        epsilon=condition.epsilon,
        projector_only=False,
    )


def _geometry_mapping(
    geometry: AdaptationGeometry | None,
    *,
    alpha: float,
    beta: float | None,
    epsilon: float,
    innovation: float | None,
    smoothed_innovation: float | None,
    distance: float,
    factor: CoordinateFactor | None,
) -> dict[str, Any]:
    return {
        "rank": 0 if geometry is None else geometry.rank,
        "eigenvalues": [] if geometry is None else [float(v) for v in geometry.eigenvalues],
        "innovation": innovation,
        "smoothed_innovation": smoothed_innovation,
        "alpha": alpha,
        "beta": beta,
        "epsilon": epsilon,
        "orthogonal_gain": (1 - alpha) + alpha * epsilon,
        "projector_distance": distance,
        "factor": {"kind": "rank_zero", "coordinate_count": 0} if factor is None else factor.mapping(),
    }


def run_trajectory(
    store: UnitStore,
    index: int,
    condition: Condition,
    *,
    phase: str,
    smoke: bool,
    resume: bool,
) -> Path:
    unit = trajectory_unit(store, index, condition, phase=phase, smoke=smoke)
    session = store.begin(unit, TRAJECTORY_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, TRAJECTORY_REQUIRED)
        assert completed is not None
        return completed
    asset_path = store.completed(
        asset_unit(store, index, phase=phase, smoke=smoke),
        ASSET_REQUIRED,
    )
    burn_path = store.completed(
        burn_unit(store, index, phase=phase, smoke=smoke),
        BURN_REQUIRED,
    )
    if asset_path is None or burn_path is None:
        raise RuntimeError("Phase 6 trajectory requires completed fresh assets and burn-in")
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    burn = torch.load(burn_path / "burn_in.pt", map_location="cpu", weights_only=False)
    plan = MixtureStreamPlan.from_mapping(assets["stream_plan"])
    if tuple(plan.p_values) != expected_schedule(phase=phase):
        raise RuntimeError("Phase 6 trajectory schedule differs from the frozen design")
    device, training_dtype, matrix_dtype = runtime(store.study)
    model, layout = build_gauge_fixed_model(
        store.study.seed(f"{phase}:trajectory_model", index),
        device=device,
        dtype=training_dtype,
    )
    model.load_state_dict(burn["model"])
    layout.assert_metadata(burn["parameter_layout"])
    fisher = representation_from_artifact(burn["fisher"], device=device)
    if not isinstance(fisher, LowRankDiagonalFisher):
        raise RuntimeError("Phase 6 burn-in Fisher is not rank plus diagonal")
    fisher = fisher.to(device=device, dtype=matrix_dtype)
    learned_geometry = _initial_geometry(
        burn["shadow_displacement"].to(device=device, dtype=matrix_dtype)
    )
    geometry = learned_geometry
    if condition.kind == "random_rank_matched" and geometry is not None:
        geometry = random_geometry(
            layout.total_numel,
            geometry.eigenvalues,
            seed=store.study.seed(f"{phase}:random_basis", index),
        )
    controller_state = InnovationControllerState()
    head_mask = _head_mask(layout, device=device, dtype=training_dtype)
    evaluation_inputs = assets["evaluation_inputs"].to(device=device, dtype=training_dtype)
    evaluation_targets = assets["evaluation_targets"].to(device=device)
    rows: list[dict[str, Any]] = []
    pr_curves: list[dict[str, Any]] = []
    parameters = [layout.flatten_module(model, detach=True).cpu()]
    displacements = []
    shadow_displacements = []
    bases: list[torch.Tensor] = []
    eigenvalues: list[torch.Tensor] = []
    learner_seconds = shadow_seconds = fisher_seconds = evaluation_seconds = 0.0
    cumulative_nines = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, len(plan.p_values)):
        p = float(plan.p_values[step])
        before = layout.flatten_module(model, detach=True)
        evaluation_started = time.perf_counter()
        evaluation, _ = evaluate_low_prevalence(
            model,
            evaluation_inputs,
            evaluation_targets,
            p,
        )
        evaluation_seconds += time.perf_counter() - evaluation_started
        inputs = assets["stream_inputs"][step].to(device=device, dtype=training_dtype)
        targets = assets["stream_targets"][step].to(device=device)
        batch_nines = int((targets == 9).sum())
        bases.append(
            torch.empty(layout.total_numel, 0)
            if geometry is None
            else geometry.basis.detach().cpu()
        )
        eigenvalues.append(
            torch.empty(0) if geometry is None else geometry.eigenvalues.detach().cpu()
        )
        fisher_started = time.perf_counter()
        fresh, _ = _fresh_fisher(
            model,
            layout,
            inputs,
            targets,
            matrix_dtype=matrix_dtype,
        )
        fisher_seconds += time.perf_counter() - fisher_started

        innovation = None
        alpha = 1.0
        beta = None
        distance = 0.0
        shadow = torch.zeros(layout.total_numel, device=device, dtype=matrix_dtype)
        optimizer_metrics = None
        shadow_metrics = None
        factor: CoordinateFactor | None = None
        if condition.kind == "no_update":
            actual = torch.zeros(layout.total_numel, device=device, dtype=training_dtype)
            alpha = 0.0
        elif condition.kind == "full_space":
            factor = _factor_for_condition(
                condition,
                layout,
                geometry,
                head_mask,
                alpha=alpha,
                device=device,
                training_dtype=training_dtype,
            )
            assert factor is not None
            learner_started = time.perf_counter()
            result = _optimize(store, model, layout, inputs, targets, fisher, factor)
            learner_seconds += time.perf_counter() - learner_started
            actual = result.displacement
            shadow = actual.to(device=device, dtype=matrix_dtype)
            optimizer_metrics = result.mapping()
            shadow_metrics = result.mapping()
        else:
            shadow_started = time.perf_counter()
            shadow_result = _optimize(
                store,
                model,
                layout,
                inputs,
                targets,
                fisher,
                identity_factor(layout.total_numel, device=device, dtype=training_dtype),
            )
            shadow_seconds += time.perf_counter() - shadow_started
            shadow = shadow_result.displacement.to(
                device=device,
                dtype=matrix_dtype,
            )
            shadow_metrics = shadow_result.mapping()
            layout.copy_vector_to_module(model, before)
            if condition.kind == "adaptive_subgd":
                assert condition.controller is not None
                if geometry is None:
                    innovation = 1.0
                    decision, controller_state = controller_state.decide(
                        innovation,
                        condition.controller,
                    )
                    alpha = 0.0
                    beta = decision.beta
                else:
                    innovation = geometry.innovation(shadow)
                    decision, controller_state = controller_state.decide(
                        innovation,
                        condition.controller,
                    )
                    alpha = decision.alpha
                    beta = decision.beta
            factor = _factor_for_condition(
                condition,
                layout,
                geometry,
                head_mask,
                alpha=alpha,
                device=device,
                training_dtype=training_dtype,
            )
            if factor is None:
                actual = torch.zeros(
                    layout.total_numel,
                    device=device,
                    dtype=training_dtype,
                )
                optimizer_metrics = {
                    "solver_parameterization": "rank_zero_no_update",
                    "objective_decrease": 0.0,
                    "displacement_norm": 0.0,
                    "factor": {"kind": "rank_zero", "coordinate_count": 0},
                }
            else:
                learner_started = time.perf_counter()
                result = _optimize(store, model, layout, inputs, targets, fisher, factor)
                learner_seconds += time.perf_counter() - learner_started
                actual = result.displacement
                optimizer_metrics = result.mapping()

        displacements.append(actual.detach().cpu())
        shadow_displacements.append(shadow.detach().cpu())
        parameters.append(layout.flatten_module(model, detach=True).cpu())
        prior_geometry = geometry
        if condition.kind == "adaptive_subgd":
            assert beta is not None
            geometry = _grow_geometry(geometry, shadow, beta)
            distance = _projector_distance(prior_geometry, geometry)

        update_started = time.perf_counter()
        update = blend_archive_fisher(
            fisher,
            fresh,
            blend_gain=MIXTURE_PI,
            rank=8,
            lanczos_seed=derive_component_seed(
                store.study.seed(f"{phase}:replica", index),
                f"trajectory_update_lanczos:{step}",
            ),
        )
        fisher = update.representation
        fisher_seconds += time.perf_counter() - update_started
        rows.append(
            {
                "step": step,
                "timing": "pre_update",
                "p": p,
                "post_burn_in_step": step - 1,
                "post_burn_in_observations": (step - 1) * plan.samples_per_step,
                "observations_before_evaluation": (step - 1) * plan.samples_per_step,
                "batch_nines": batch_nines,
                "cumulative_observed_nines": cumulative_nines,
                "schedule_kind": "low_prevalence_linear",
                "condition": condition.name,
                "parameter_hash": tensor_content_hash(before.cpu()),
                "archive_trace": float(fisher.diagonal_vector().sum()),
                "optimizer": optimizer_metrics,
                "shadow": (
                    None
                    if condition.kind == "no_update"
                    else {
                        "displacement_norm": float(torch.linalg.vector_norm(shadow)),
                        "optimizer": shadow_metrics,
                    }
                ),
                "geometry": _geometry_mapping(
                    prior_geometry,
                    alpha=alpha,
                    beta=beta,
                    epsilon=condition.epsilon,
                    innovation=innovation,
                    smoothed_innovation=controller_state.smoothed_innovation,
                    distance=distance,
                    factor=factor,
                ),
                **evaluation,
            }
        )
        cumulative_nines += batch_nines
        if step in {1, 5, 10}:
            evaluation_started = time.perf_counter()
            _, curve = evaluate_low_prevalence(
                model,
                evaluation_inputs,
                evaluation_targets,
                p,
                include_pr_curve=True,
            )
            evaluation_seconds += time.perf_counter() - evaluation_started
            assert curve is not None
            pr_curves.append(
                {
                    "step": step,
                    "p": p,
                    "timing": "post_update",
                    "post_burn_in_observations": step * plan.samples_per_step,
                    **curve,
                }
            )

    final_p = float(plan.p_values[-1])
    evaluation_started = time.perf_counter()
    final_evaluation, _ = evaluate_low_prevalence(
        model,
        evaluation_inputs,
        evaluation_targets,
        final_p,
    )
    evaluation_seconds += time.perf_counter() - evaluation_started
    rows.append(
        {
            "step": len(plan.p_values) - 1,
            "timing": "post_final_update",
            "p": final_p,
            "post_burn_in_step": len(plan.p_values) - 1,
            "post_burn_in_observations": (len(plan.p_values) - 1) * plan.samples_per_step,
            "observations_before_evaluation": (len(plan.p_values) - 1) * plan.samples_per_step,
            "batch_nines": None,
            "cumulative_observed_nines": cumulative_nines,
            "schedule_kind": "low_prevalence_linear",
            "condition": condition.name,
            "parameter_hash": tensor_content_hash(parameters[-1]),
            "archive_trace": float(fisher.diagonal_vector().sum()),
            "optimizer": None,
            "shadow": None,
            "geometry": _geometry_mapping(
                geometry,
                alpha=0.0,
                beta=None,
                epsilon=condition.epsilon,
                innovation=None,
                smoothed_innovation=controller_state.smoothed_innovation,
                distance=0.0,
                factor=None,
            ),
            **final_evaluation,
        }
    )
    parameter_tensor = torch.stack(parameters)
    displacement_tensor = torch.stack(displacements)
    if not torch.equal(parameter_tensor[1:] - parameter_tensor[:-1], displacement_tensor):
        raise RuntimeError("Phase 6 trajectory displacement identity failed")
    auc_fields = (
        "precision_ref_0.1",
        "nine_ovr_recall",
        "nine_ovr_specificity",
        "false_positives_per_1000",
        "current_nll",
        "current_accuracy",
        "p0_nll",
        "p0_accuracy",
        "worst_panel_nll",
        "nine_brier_ref_0.1",
        "nine_ece_ref_0.1",
    )
    pr_by_p = {f"{row['p']:.2f}": row for row in pr_curves}
    summary = {
        "schema_version": _schema_version(phase),
        "phase": phase,
        "environment": "digit9_mixture",
        "replica_index": index,
        "schedule_kind": "low_prevalence_linear",
        "condition": condition.mapping(),
        "burn_in_steps": BURN_IN_STEPS,
        "reference_prevalence": REFERENCE_PREVALENCE,
        "realized_treatment_nines": cumulative_nines,
        "initial_basis_rank": 0 if learned_geometry is None else learned_geometry.rank,
        "final_basis_rank": 0 if geometry is None else geometry.rank,
        "source_asset_sha256": file_hash(asset_path / "assets.pt"),
        **{f"post_burn_in_{field}_auc": _auc(rows, field) for field in auc_fields},
        **{
            f"average_precision_ref_0.1_at_p_{name}": float(value["average_precision"])
            for name, value in pr_by_p.items()
        },
        "learner_wall_seconds": learner_seconds,
        "shadow_wall_seconds": shadow_seconds,
        "fisher_wall_seconds": fisher_seconds,
        "evaluation_wall_seconds": evaluation_seconds,
        "total_wall_time_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "peak_cuda_memory_bytes": (
            0 if device.type != "cuda" else int(torch.cuda.max_memory_allocated(device))
        ),
    }
    checks = {
        "all_finite": _finite_tree(rows) and _finite_tree(pr_curves) and _finite_tree(summary),
        "displacement_identity": True,
        "burn_in_pairing": torch.equal(burn["parameters"][-1], parameter_tensor[0]),
        "schedule_exact": tuple(plan.p_values) == expected_schedule(phase=phase),
        "samples_per_step_frozen": plan.samples_per_step
        == low_prevalence_data_config(smoke=smoke, phase=phase).samples_per_step,
        "expected_treatment_nines_frozen": math.isclose(
            plan.samples_per_step * sum(plan.p_values[1:]),
            _expected_treatment_nines(phase=phase, smoke=smoke),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "treatment_starts_at_p_0.01": math.isclose(
            float(rows[0]["p"]),
            0.01,
            rel_tol=0.0,
            abs_tol=8 * torch.finfo(torch.float64).eps,
        ),
        "treatment_observations": int(rows[-1]["post_burn_in_observations"])
        == 10 * plan.samples_per_step,
        "reference_prevalence_is_0.1": REFERENCE_PREVALENCE == 0.1,
        "source_asset_integrity_verified": True,
        "phase6_burn_in_loaded": True,
        "phase5_inputs_loaded": False,
        "phase5r_inputs_loaded": False,
        "cuda_exercised": device.type == "cuda",
    }
    required_true = {
        key: value
        for key, value in checks.items()
        if key not in {"phase5_inputs_loaded", "phase5r_inputs_loaded", "cuda_exercised"}
    }
    if not all(required_true.values()):
        raise RuntimeError(f"Phase 6 trajectory checks failed: {checks}")
    session.write_torch(
        "trajectory.pt",
        {
            "parameters": parameter_tensor,
            "displacements": displacement_tensor,
            "shadow_displacements": torch.stack(shadow_displacements),
            "bases": bases,
            "eigenvalues": eigenvalues,
        },
    )
    session.write_json("metrics.json", rows)
    session.write_json("pr_curves.json", pr_curves)
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    session.write_torch(
        "final_state.pt",
        {
            "model": _state_dict_cpu(model),
            "fisher": fisher.artifact_mapping(),
            "geometry_basis": (
                torch.empty(layout.total_numel, 0)
                if geometry is None
                else geometry.basis.detach().cpu()
            ),
            "geometry_eigenvalues": (
                torch.empty(0) if geometry is None else geometry.eigenvalues.detach().cpu()
            ),
            "controller_state": dataclasses.asdict(controller_state),
        },
    )
    return store.finish(session, TRAJECTORY_REQUIRED)


def run_analysis(
    store: UnitStore,
    replicas: tuple[int, ...],
    *,
    phase: str,
    smoke: bool,
    resume: bool,
) -> Path:
    if not smoke and replicas != tuple(range(1, PRODUCTION_REPLICAS + 1)):
        raise ValueError("Phase 6 production analysis requires the frozen 64-replica cohort")
    unit = analysis_unit(store, replicas, phase=phase, smoke=smoke)
    session = store.begin(unit, ANALYSIS_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ANALYSIS_REQUIRED)
        assert completed is not None
        return completed
    conditions = _conditions(phase)
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trajectory_hashes = []
    for index in replicas:
        for condition in conditions:
            path = store.completed(
                trajectory_unit(
                    store,
                    index,
                    condition,
                    phase=phase,
                    smoke=smoke,
                ),
                TRAJECTORY_REQUIRED,
            )
            if path is None:
                raise RuntimeError(f"missing Phase 6 trajectory {index}/{condition.name}")
            by_condition[condition.name].append(_read_json(path / "summary.json"))
            trajectory_hashes.append(file_hash(path / "summary.json"))
    fields = {
        "precision_ref_0.1": True,
        "nine_ovr_recall": True,
        "nine_ovr_specificity": True,
        "false_positives_per_1000": False,
        "current_nll": False,
        "current_accuracy": True,
        "p0_nll": False,
        "p0_accuracy": True,
        "worst_panel_nll": False,
        "nine_brier_ref_0.1": False,
        "nine_ece_ref_0.1": False,
    }
    aggregate = []
    for condition in conditions:
        values = by_condition[condition.name]
        aggregate.append(
            {
                "condition": condition.name,
                "replicas": len(values),
                **{
                    f"mean_{field}_auc": statistics.fmean(
                        float(value[f"post_burn_in_{field}_auc"]) for value in values
                    )
                    for field in fields
                },
                "mean_realized_treatment_nines": statistics.fmean(
                    float(value["realized_treatment_nines"]) for value in values
                ),
                "mean_final_basis_rank": statistics.fmean(
                    float(value["final_basis_rank"]) for value in values
                ),
                "mean_wall_seconds": statistics.fmean(
                    float(value["total_wall_time_seconds"]) for value in values
                ),
            }
        )

    def comparison(name: str, method_name: str, comparator_name: str) -> dict[str, Any]:
        method = by_condition[method_name]
        comparator = by_condition[comparator_name]
        return {
            "name": name,
            "treatment": method_name,
            "comparator": comparator_name,
            **{
                field: _paired_effect(
                    [float(value[f"post_burn_in_{field}_auc"]) for value in method],
                    [float(value[f"post_burn_in_{field}_auc"]) for value in comparator],
                    higher_is_better=higher,
                )
                for field, higher in fields.items()
            },
        }

    comparisons = [
        comparison("primary_head_vs_full", "head_only", "full_space"),
        comparison(
            "subgd_adaptive_vs_full",
            "adaptive_floor_0.1",
            "full_space",
        ),
        comparison(
            "practical_head_vs_adaptive",
            "head_only",
            "adaptive_floor_0.1",
        ),
        comparison(
            "dimensional_bias_vs_head",
            "digit9_bias_only",
            "head_only",
        ),
    ]
    if _is_scaling_phase(phase):
        comparisons.insert(
            1,
            comparison("absolute_head_vs_no_update", "head_only", "no_update"),
        )
    else:
        comparisons.append(
            comparison(
                "mechanistic_static_vs_random",
                "static_tiny_burn_subgd",
                "random_rank_one",
            )
        )
    summary = {
        "schema_version": _schema_version(phase),
        "phase": phase,
        "environment": "digit9_mixture",
        "classification": "smoke_not_evidence"
        if smoke
        else (
            "exploratory_hundred_positive_closeout"
            if _is_phase6b(phase)
            else (
                "exploratory_ten_positive_development"
                if _is_phase6a(phase)
                else "prospective_low_prevalence_production"
            )
        ),
        "reference_prevalence": REFERENCE_PREVALENCE,
        "replicas": list(replicas),
        "aggregate": aggregate,
        "comparisons": comparisons,
        "trajectory_summary_hashes": trajectory_hashes,
        "prior_mixture_evidence_pooled": False,
        "complete_fixed_size_inference": (not smoke and len(replicas) == PRODUCTION_REPLICAS),
    }
    checks = {
        "all_finite": _finite_tree(summary),
        "replica_count": len(replicas),
        "condition_count": len(conditions),
        "trajectory_count": len(trajectory_hashes),
        "all_trajectories_present": len(trajectory_hashes) == len(replicas) * len(conditions),
        "prior_mixture_evidence_pooled": False,
        "analysis_hash": canonical_hash(summary),
    }
    session.write_json("summary.json", summary)
    session.write_json("checks.json", checks)
    return store.finish(session, ANALYSIS_REQUIRED)


def build_ledger(
    store: UnitStore,
    replicas: tuple[int, ...],
    *,
    phase: str,
    smoke: bool,
) -> dict[str, Any]:
    items = []
    for index in replicas:
        items.append(
            {
                "action": "phase6_assets",
                "unit": asset_unit(store, index, phase=phase, smoke=smoke),
                "required": list(ASSET_REQUIRED),
                "parameters": {"index": index},
            }
        )
        items.append(
            {
                "action": "phase6_burn_in",
                "unit": burn_unit(store, index, phase=phase, smoke=smoke),
                "required": list(BURN_REQUIRED),
                "parameters": {"index": index},
            }
        )
        for condition in _conditions(phase):
            items.append(
                {
                    "action": "phase6_trajectory",
                    "unit": trajectory_unit(
                        store,
                        index,
                        condition,
                        phase=phase,
                        smoke=smoke,
                    ),
                    "required": list(TRAJECTORY_REQUIRED),
                    "parameters": {
                        "index": index,
                        "condition": condition.mapping(),
                    },
                }
            )
    items.append(
        {
            "action": "phase6_analysis",
            "unit": analysis_unit(store, replicas, phase=phase, smoke=smoke),
            "required": list(ANALYSIS_REQUIRED),
            "parameters": {},
        }
    )
    return {
        "study_hash": store.study.config_hash,
        "source_hashes": store.sources,
        "phase": phase,
        "schema_version": _schema_version(phase),
        "smoke": smoke,
        "replicas": list(replicas),
        "items": items,
    }


def _execute_item(
    store: UnitStore,
    replicas: tuple[int, ...],
    item: dict[str, Any],
    *,
    phase: str,
    smoke: bool,
    data_root: Path,
    resume: bool,
) -> Path:
    action = item["action"]
    index = int(item["parameters"].get("index", 0))
    if action == "phase6_assets":
        return ensure_assets(
            store,
            index,
            phase=phase,
            smoke=smoke,
            data_root=data_root,
            resume=resume,
        )
    if action == "phase6_burn_in":
        return run_burn_in(
            store,
            index,
            phase=phase,
            smoke=smoke,
            data_root=data_root,
            resume=resume,
        )
    if action == "phase6_trajectory":
        return run_trajectory(
            store,
            index,
            condition_from_mapping(item["parameters"]["condition"]),
            phase=phase,
            smoke=smoke,
            resume=resume,
        )
    if action == "phase6_analysis":
        return run_analysis(
            store,
            replicas,
            phase=phase,
            smoke=smoke,
            resume=resume,
        )
    raise RuntimeError(f"unsupported Phase 6 ledger action: {action}")


def run(
    store: UnitStore,
    *,
    phase: str,
    smoke: bool,
    replicas: tuple[int, ...],
    data_root: Path,
    notebook: Path,
    resume: bool,
    max_units: int | None,
    max_wall_seconds: float | None,
) -> tuple[int, bool]:
    ledger = build_ledger(store, replicas, phase=phase, smoke=smoke)
    ledger_name = f"{phase}_{canonical_hash(ledger)[:12]}.json" if smoke else f"{phase}.json"
    freeze_json(store.root / "ledgers" / ledger_name, ledger, resume=resume)
    if not smoke:
        refresh(store, notebook)
    started = time.monotonic()
    last_refresh = started
    completed_now = 0
    exhausted = False
    for item in ledger["items"]:
        required = tuple(item["required"])
        if store.completed(item["unit"], required) is not None:
            continue
        if max_units is not None and completed_now >= max_units:
            exhausted = True
            break
        if max_wall_seconds is not None and time.monotonic() - started >= max_wall_seconds:
            exhausted = True
            break
        try:
            path = _execute_item(
                store,
                replicas,
                item,
                phase=phase,
                smoke=smoke,
                data_root=data_root,
                resume=resume,
            )
        except BaseException as error:
            store.record_failure(item["unit"], error)
            if not smoke:
                refresh(store, notebook)
            raise
        completed_now += 1
        print(f"[phase6] completed {item['action']}: {path}", flush=True)
        if not smoke and (
            item["action"] == "phase6_analysis"
            or time.monotonic() - last_refresh >= 300
        ):
            refresh(store, notebook)
            last_refresh = time.monotonic()
    if not smoke:
        refresh(store, notebook)
    return completed_now, exhausted


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cuda-smoke", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--max-wall-seconds", type=float)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.smoke and arguments.cuda_smoke:
        raise ValueError("choose either --smoke or --cuda-smoke")
    if arguments.max_units is not None and arguments.max_units < 1:
        raise ValueError("--max-units must be positive")
    if arguments.max_wall_seconds is not None and arguments.max_wall_seconds <= 0:
        raise ValueError("--max-wall-seconds must be positive")
    repo_root = Path(__file__).parents[2]
    study = Plan13Study.from_path(arguments.config)
    store = UnitStore(arguments.output_root, study, repo_root)
    smoke = arguments.smoke or arguments.cuda_smoke
    phase = (
        CUDA_SMOKE_PHASE
        if arguments.cuda_smoke
        else SMOKE_PHASE if arguments.smoke else PHASE
    )
    replicas = (1,) if smoke else tuple(range(1, PRODUCTION_REPLICAS + 1))
    completed, exhausted = run(
        store,
        phase=phase,
        smoke=smoke,
        replicas=replicas,
        data_root=arguments.data_root,
        notebook=arguments.notebook,
        resume=arguments.resume,
        max_units=arguments.max_units,
        max_wall_seconds=arguments.max_wall_seconds,
    )
    print(f"[phase6] completed_now={completed} exhausted={exhausted}")


if __name__ == "__main__":
    main()


__all__ = [
    "ANALYSIS_REQUIRED",
    "ASSET_REQUIRED",
    "BURN_REQUIRED",
    "PHASE",
    "PHASE6A",
    "PHASE6A_CUDA_SMOKE",
    "PHASE6A_SCHEMA_VERSION",
    "PHASE6A_SMOKE",
    "PHASE6B",
    "PHASE6B_CUDA_SMOKE",
    "PHASE6B_SCHEMA_VERSION",
    "PHASE6B_SMOKE",
    "PRODUCTION_REPLICAS",
    "SCHEMA_VERSION",
    "TRAJECTORY_REQUIRED",
    "analysis_unit",
    "asset_unit",
    "build_ledger",
    "burn_unit",
    "digit9_bias_factor",
    "digit9_bias_functional_error",
    "evaluate_low_prevalence",
    "expected_schedule",
    "fixed_prevalence_precision",
    "low_prevalence_data_config",
    "phase6_conditions",
    "phase6a_conditions",
    "phase6b_conditions",
    "run_analysis",
    "run_burn_in",
    "run_trajectory",
    "standardized_precision_recall_curve",
    "trajectory_unit",
]
