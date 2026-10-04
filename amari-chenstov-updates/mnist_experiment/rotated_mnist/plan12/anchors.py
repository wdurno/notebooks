"""Fresh fixed-pi paths and frozen local-response anchors for Plan 12."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.ewc import build_optimizer, take_ewc_proposal
from src.hybrid import blend_archive_fisher
from src.mnist_data import load_mnist_datasets
from src.parameters import ParameterLayout
from src.representations import LowRankDiagonalFisher
from src.seeding import derive_component_seed

from ..data import RotatedPartitions, RotatedStreamPlan
from ..run import _learner_optimizer_config, _state_dict_cpu
from ..run_phase3 import _fresh_fisher
from ..transform import rotate_mnist_batch, tensor_content_hash
from .artifacts import UnitStore
from .assets import ensure_replica_assets, runtime
from .config import PI
from .trajectory import TrajectoryCondition, _load_model_and_fishers


ANCHOR_REQUIRED = ("anchors.pt", "summary.json")


def anchor_steps(num_transitions: int, count: int) -> tuple[int, ...]:
    if count == 1:
        return (num_transitions // 2,)
    if count != 4 or num_transitions % 6:
        raise ValueError("the frozen four-anchor design requires six-divisible transitions")
    return (0, num_transitions // 6, num_transitions // 2, 5 * num_transitions // 6)


def _materialize_indices(dataset: Any, indices: tuple[int, ...]) -> tuple[Tensor, Tensor]:
    examples = [dataset[index] for index in indices]
    return torch.stack([item[0] for item in examples]), torch.as_tensor([int(item[1]) for item in examples])


def _local_samples(
    assets: dict[str, Any],
    anchor_index: int,
    angle: float,
    store: UnitStore,
    train_dataset: Any,
) -> dict[str, Tensor]:
    partitions = RotatedPartitions.from_mapping(assets["partitions"])
    needed = store.study.local_target_samples + 4 * store.study.local_batches_per_anchor
    if needed > len(partitions.reference):
        raise RuntimeError("local target and branch samples exceed the reference pool")
    generator = torch.Generator().manual_seed(store.study.seed("phase1:local-samples", anchor_index))
    order = torch.randperm(len(partitions.reference), generator=generator)[:needed]
    indices = tuple(partitions.reference[position] for position in order.tolist())
    inputs, targets = _materialize_indices(train_dataset, indices)
    inputs = rotate_mnist_batch(inputs, angle, store.study.protocol.rotation)
    target_count = store.study.local_target_samples
    evaluation_count = min(256 if store.study.smoke else 2048, assets["base_inputs"].shape[0])
    evaluation_inputs = assets["base_inputs"][:evaluation_count]
    evaluation_targets = assets["base_targets"][:evaluation_count]
    return {
        "nine_prevalence": assets["nine_prevalence"],
        "population_inputs": inputs[:target_count],
        "population_targets": targets[:target_count],
        "batch_inputs": inputs[target_count:].reshape(store.study.local_batches_per_anchor, 4, 1, 28, 28),
        "batch_targets": targets[target_count:].reshape(store.study.local_batches_per_anchor, 4),
        "current_evaluation_inputs": rotate_mnist_batch(
            evaluation_inputs,
            angle,
            store.study.protocol.rotation,
        ),
        "evaluation_targets": evaluation_targets,
        "retention_000_inputs": rotate_mnist_batch(evaluation_inputs, 0.0, store.study.protocol.rotation),
        "retention_015_inputs": rotate_mnist_batch(evaluation_inputs, 15.0, store.study.protocol.rotation),
        "retention_030_inputs": rotate_mnist_batch(evaluation_inputs, 30.0, store.study.protocol.rotation),
        "sample_indices": torch.as_tensor(indices, dtype=torch.long),
    }


def _advance(
    model: nn.Module,
    layout: ParameterLayout,
    optimizer: torch.optim.Optimizer,
    fisher: LowRankDiagonalFisher,
    dense_archive: Tensor,
    inputs: Tensor,
    targets: Tensor,
    *,
    config: Any,
    seed_label: str,
    device: torch.device,
    training_dtype: torch.dtype,
    matrix_dtype: torch.dtype,
) -> tuple[LowRankDiagonalFisher, Tensor, dict[str, Any]]:
    anchor = layout.flatten_module(model, detach=True)
    fresh = _fresh_fisher(model, layout, inputs, targets, matrix_dtype=matrix_dtype)
    proposal = take_ewc_proposal(
        model,
        layout,
        inputs,
        targets,
        fisher.to(dtype=training_dtype),
        _learner_optimizer_config(config),
        optimizer,
        adaptation_weight=PI,
        penalty_anchor=anchor,
    )
    dense = (1 - PI) * dense_archive + PI * fresh
    dense = (dense + dense.mT) / 2
    update = blend_archive_fisher(
        fisher,
        fresh,
        blend_gain=PI,
        rank=config.fisher.rank,
        lanczos_seed=derive_component_seed(config.replica_seed, seed_label),
    )
    return update.representation, dense, {
        "proposal": proposal.metrics_mapping(),
        "lanczos": update.lanczos.mapping(),
    }


def ensure_anchor_path(
    store: UnitStore,
    schedule_kind: str,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    asset_path = ensure_replica_assets(store, "phase1", 1, data_root=data_root, resume=True)
    unit = store.unit("phase1", "anchors", 1, schedule=schedule_kind)
    session = store.begin(unit, ANCHOR_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ANCHOR_REQUIRED)
        assert completed is not None
        return completed
    started = time.perf_counter()
    study = store.study
    config = study.protocol_for_replica("phase1", 1)
    device, training_dtype, matrix_dtype = runtime(study)
    assets = torch.load(asset_path / "assets.pt", map_location="cpu", weights_only=False)
    stream_plan = RotatedStreamPlan.from_mapping(assets["stream_plans"][schedule_kind])
    schedule = stream_plan.schedule
    selected = anchor_steps(schedule.num_transitions, study.anchors_per_schedule)
    stream = assets["streams"][schedule_kind]
    train_dataset, _ = load_mnist_datasets(data_root, download=False)

    chart_condition = TrajectoryCondition("gauge_no_ridge", chart=True)
    raw_condition = TrajectoryCondition("legacy_raw", chart=False)
    chart_model, chart_layout, chart_fisher, chart_dense = _load_model_and_fishers(
        assets,
        chart_condition,
        seed=config.replica_seed,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
    )
    raw_model, raw_layout, raw_fisher, raw_dense = _load_model_and_fishers(
        assets,
        raw_condition,
        seed=config.replica_seed,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
    )
    chart_optimizer = build_optimizer(chart_model, _learner_optimizer_config(config))
    raw_optimizer = build_optimizer(raw_model, _learner_optimizer_config(config))
    anchors: list[dict[str, Any]] = []
    update_health: list[dict[str, Any]] = []
    schedule_offset = 0 if schedule_kind == "linear" else study.anchors_per_schedule

    for step in range(schedule.num_points):
        if step in selected:
            ordinal = selected.index(step)
            anchor_index = schedule_offset + ordinal + 1
            samples = _local_samples(
                assets,
                anchor_index,
                float(schedule.angles_degrees[step]),
                store,
                train_dataset,
            )
            anchors.append(
                {
                    "anchor_index": anchor_index,
                    "anchor_ordinal": ordinal,
                    "step": step,
                    "angle_degrees": float(schedule.angles_degrees[step]),
                    "leg_id": schedule.leg_ids[step],
                    "chart_state": _state_dict_cpu(chart_model),
                    "chart_parameter": chart_layout.flatten_module(chart_model, detach=True).cpu(),
                    "chart_fisher": chart_fisher.artifact_mapping(),
                    "chart_dense_archive": chart_dense.detach().cpu(),
                    "raw_state": _state_dict_cpu(raw_model),
                    "raw_parameter": raw_layout.flatten_module(raw_model, detach=True).cpu(),
                    "raw_fisher": raw_fisher.artifact_mapping(),
                    "raw_dense_archive": raw_dense.detach().cpu(),
                    "samples": samples,
                }
            )
        if step == schedule.num_transitions:
            break
        inputs = stream["inputs"][step].to(device=device, dtype=training_dtype)
        targets = stream["targets"][step].to(device=device)
        chart_fisher, chart_dense, chart_health = _advance(
            chart_model,
            chart_layout,
            chart_optimizer,
            chart_fisher,
            chart_dense,
            inputs,
            targets,
            config=config,
            seed_label=f"plan12_anchor_chart:{schedule_kind}:step={step}",
            device=device,
            training_dtype=training_dtype,
            matrix_dtype=matrix_dtype,
        )
        raw_fisher, raw_dense, raw_health = _advance(
            raw_model,
            raw_layout,
            raw_optimizer,
            raw_fisher,
            raw_dense,
            inputs,
            targets,
            config=config,
            seed_label=f"plan12_anchor_raw:{schedule_kind}:step={step}",
            device=device,
            training_dtype=training_dtype,
            matrix_dtype=matrix_dtype,
        )
        update_health.append({"step": step, "chart": chart_health, "raw": raw_health})

    if len(anchors) != study.anchors_per_schedule:
        raise RuntimeError("Plan 12 anchor path did not produce its frozen anchor count")
    summary = {
        "schedule_kind": schedule_kind,
        "selected_steps": list(selected),
        "anchor_indices": [anchor["anchor_index"] for anchor in anchors],
        "sample_hashes": [tensor_content_hash(anchor["samples"]["sample_indices"]) for anchor in anchors],
        "wall_time_seconds": time.perf_counter() - started,
        "update_health": update_health,
    }
    session.write_torch("anchors.pt", anchors)
    session.write_json("summary.json", summary)
    return store.finish(session, ANCHOR_REQUIRED)


def load_anchor(path: Path, anchor_index: int) -> dict[str, Any]:
    anchors = torch.load(path / "anchors.pt", map_location="cpu", weights_only=False)
    matches = [anchor for anchor in anchors if anchor["anchor_index"] == anchor_index]
    if len(matches) != 1:
        raise RuntimeError(f"expected one Plan 12 anchor {anchor_index}, found {len(matches)}")
    return matches[0]
