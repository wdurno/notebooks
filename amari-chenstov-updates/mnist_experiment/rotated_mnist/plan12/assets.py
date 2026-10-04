"""Paired initialization, stream, panel, and Fisher assets for Plan 12."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from src.initialization import state_dict_hash
from src.lanczos_wrapper import approximate_low_rank_diagonal
from src.mnist_data import dataset_targets, load_mnist_datasets
from src.mnist_model import build_canonical_model, configure_torch_runtime, resolve_device, resolve_dtype
from src.seeding import derive_component_seed

from ..data import generate_rotated_stream, partition_all_digit_mnist
from ..phase4_metrics import materialize_base_panel
from ..run import _fit_upright_initializer, _state_dict_cpu
from ..run_phase3 import _estimate_initial_fisher
from ..schedule import resolve_shaped_rotation_schedule
from ..transform import tensor_content_hash
from .artifacts import UnitStore
from .gauge import build_gauge_fixed_model, chart_embedding, load_canonical_state, raw_fisher_to_chart


ASSET_REQUIRED = ("assets.pt", "summary.json")


def runtime(study: Any) -> tuple[torch.device, torch.dtype, torch.dtype]:
    config = study.protocol
    device = resolve_device(config.runtime.device)
    training_dtype = resolve_dtype(config.runtime.dtype)
    matrix_dtype = resolve_dtype(config.fisher.matrix_dtype)
    configure_torch_runtime(
        deterministic_algorithms=config.runtime.deterministic_algorithms,
        warn_only=device.type == "cuda",
    )
    return device, training_dtype, matrix_dtype


def ensure_replica_assets(
    store: UnitStore,
    phase: str,
    index: int,
    *,
    data_root: Path,
    resume: bool,
) -> Path:
    unit = store.unit(phase, "assets", index)
    session = store.begin(unit, ASSET_REQUIRED, resume=resume)
    if session is None:
        completed = store.completed(unit, ASSET_REQUIRED)
        assert completed is not None
        return completed
    study = store.study
    config = study.protocol_for_replica(phase, index)
    device, training_dtype, matrix_dtype = runtime(study)
    started = time.perf_counter()
    train_dataset, test_dataset = load_mnist_datasets(data_root, download=False)
    train_targets = dataset_targets(train_dataset)
    test_targets = dataset_targets(test_dataset)
    nine_prevalence = float((test_targets == 9).double().mean())
    partitions = partition_all_digit_mnist(
        train_targets,
        test_targets,
        config.data,
        replica_seed=config.replica_seed,
    )
    schedules = {
        kind: resolve_shaped_rotation_schedule(
            config.rotation,
            kind=kind,
            sigmoid_kappa=config.sigmoid_kappa,
        )
        for kind in config.schedule_kinds
    }
    streams = {
        kind: generate_rotated_stream(
            train_dataset,
            train_targets,
            partitions,
            schedules[kind],
            config.data,
            config.rotation,
            replica_seed=config.replica_seed,
        )
        for kind in config.schedule_kinds
    }
    linear, sigmoid = streams["linear"][0], streams["sigmoid"][0]
    if linear.observation_indices != sigmoid.observation_indices or linear.class_labels != sigmoid.class_labels:
        raise RuntimeError("Plan 12 linear and sigmoid stream identities differ")

    raw_model, raw_layout = build_canonical_model(
        derive_component_seed(config.replica_seed, "plan5_model_initialization"),
        device=device,
        dtype=training_dtype,
    )
    initialization = _fit_upright_initializer(
        raw_model,
        train_dataset,
        test_dataset,
        partitions.initialization,
        partitions.evaluation,
        config,
        nine_prevalence=nine_prevalence,
        device=device,
        dtype=training_dtype,
    )
    raw_state = _state_dict_cpu(raw_model)
    raw_dense, fisher_metrics = _estimate_initial_fisher(
        raw_model,
        raw_layout,
        train_dataset,
        partitions.reference,
        config,
        device=device,
        training_dtype=training_dtype,
        matrix_dtype=matrix_dtype,
    )
    raw_approximation = approximate_low_rank_diagonal(
        lambda vector: raw_dense @ vector,
        torch.diagonal(raw_dense),
        rank=config.fisher.rank,
        seed=derive_component_seed(config.replica_seed, "plan12_initial_lanczos:raw"),
    )
    chart_model, chart_layout = build_gauge_fixed_model(
        config.replica_seed,
        device=device,
        dtype=training_dtype,
    )
    load_canonical_state(chart_model, raw_model)
    embedding = chart_embedding(
        raw_layout,
        chart_layout,
        device=device,
        dtype=matrix_dtype,
    )
    chart_dense = raw_fisher_to_chart(raw_dense, embedding)
    chart_approximation = approximate_low_rank_diagonal(
        lambda vector: chart_dense @ vector,
        torch.diagonal(chart_dense),
        rank=config.fisher.rank,
        seed=derive_component_seed(config.replica_seed, "plan12_initial_lanczos:chart"),
    )
    base_inputs, base_targets = materialize_base_panel(
        test_dataset,
        partitions.evaluation,
        num_workers=config.runtime.num_workers,
    )
    assets = {
        "partitions": partitions.to_mapping(),
        "stream_plans": {kind: streams[kind][0].to_mapping() for kind in streams},
        "streams": {
            kind: {"inputs": streams[kind][1], "targets": streams[kind][2]}
            for kind in streams
        },
        "raw_initial_state": raw_state,
        "chart_initial_state": _state_dict_cpu(chart_model),
        "raw_initial_dense_fisher": raw_dense.detach().cpu(),
        "chart_initial_dense_fisher": chart_dense.detach().cpu(),
        "raw_initial_fisher": raw_approximation.representation.artifact_mapping(),
        "chart_initial_fisher": chart_approximation.representation.artifact_mapping(),
        "raw_parameter_layout": raw_layout.metadata(),
        "chart_parameter_layout": chart_layout.metadata(),
        "base_inputs": base_inputs,
        "base_targets": base_targets,
        "nine_prevalence": nine_prevalence,
    }
    summary = {
        "replica_seed": config.replica_seed,
        "initialization": initialization,
        "initial_model_state_hash": state_dict_hash(raw_state),
        "raw_initial_fisher": {
            **fisher_metrics,
            "lanczos": raw_approximation.diagnostics.mapping(),
        },
        "chart_initial_fisher": {
            "trace": float(torch.trace(chart_dense)),
            "frobenius_norm": float(torch.linalg.matrix_norm(chart_dense, ord="fro")),
            "lanczos": chart_approximation.diagnostics.mapping(),
        },
        "raw_parameter_count": raw_layout.total_numel,
        "chart_parameter_count": chart_layout.total_numel,
        "evaluation_inputs_hash": tensor_content_hash(base_inputs),
        "evaluation_targets_hash": tensor_content_hash(base_targets),
        "stream_identity_paired": True,
        "stream_hashes": {
            kind: tensor_content_hash(streams[kind][1]) for kind in streams
        },
        "wall_time_seconds": time.perf_counter() - started,
    }
    session.write_torch("assets.pt", assets)
    session.write_json("summary.json", summary)
    return store.finish(session, ASSET_REQUIRED)
