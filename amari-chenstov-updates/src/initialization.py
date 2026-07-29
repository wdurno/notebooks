"""Fit and persist treatment-independent MNIST replica initialization bundles."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
from functools import partial
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from .artifacts import collect_runtime_metadata
from .config import ExperimentConfig, InitializationConfig
from .mnist_data import (
    MNIST_DATA_SCHEMA_VERSION,
    STREAM_PLAN_SCHEMA_VERSION,
    DatasetPartitions,
    MixtureStreamPlan,
)
from .mnist_model import (
    MNIST_MODEL_SCHEMA_VERSION,
    CanonicalMnistCNN,
    build_canonical_model,
    resolve_dtype,
)
from .parameters import ParameterLayout
from .seeding import derive_seed_map

REPLICA_BUNDLE_SCHEMA_VERSION = 2


class ReplicaBundleError(RuntimeError):
    """Raised when a shared replica bundle is incomplete or incompatible."""


@dataclasses.dataclass(frozen=True)
class InitializationResult:
    started_at: str
    completed_at: str
    wall_time_seconds: float
    epochs_completed: int
    stopped_on_target: bool
    history: tuple[dict[str, float | int], ...]
    final_metrics: dict[str, float | int]

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class LoadedReplicaBundle:
    model: CanonicalMnistCNN
    layout: ParameterLayout
    partitions: DatasetPartitions
    stream_plan: MixtureStreamPlan
    initialization: InitializationResult
    metadata: dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def replica_design_mapping(config: ExperimentConfig) -> dict[str, Any]:
    """Return treatment-independent inputs shared by all paired conditions."""

    return {
        "bundle_schema_version": REPLICA_BUNDLE_SCHEMA_VERSION,
        "model_schema_version": MNIST_MODEL_SCHEMA_VERSION,
        "data_schema_version": MNIST_DATA_SCHEMA_VERSION,
        "stream_plan_schema_version": STREAM_PLAN_SCHEMA_VERSION,
        "replica_id": config.replica_id,
        "replica_seed": config.replica_seed,
        "data": dataclasses.asdict(config.data),
        "initialization": dataclasses.asdict(config.initialization),
        "training_dtype": config.runtime.training_dtype,
        "deterministic_algorithms": config.runtime.deterministic_algorithms,
        "seeds": derive_seed_map(config.replica_seed),
    }


def replica_design_hash(config: ExperimentConfig) -> str:
    return hashlib.sha256(
        _canonical_json(replica_design_mapping(config)).encode("utf-8")
    ).hexdigest()


def replica_bundle_id(config: ExperimentConfig) -> str:
    return f"{config.replica_id}__{replica_design_hash(config)[:16]}"


def _seed_worker(worker_id: int, base_seed: int) -> None:
    worker_seed = (base_seed + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        worker_init_fn=partial(_seed_worker, base_seed=seed),
        persistent_workers=num_workers > 0,
    )


def evaluate_classifier(
    model: nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    num_workers: int = 0,
) -> dict[str, float | int]:
    loader = _make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
    )
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    group_loss = {False: 0.0, True: 0.0}
    group_count = {False: 0, True: 0}
    group_correct = {False: 0, True: 0}
    class_count = torch.zeros(10, dtype=torch.long)
    class_correct = torch.zeros(10, dtype=torch.long)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=dtype)
            targets = targets.to(device=device)
            logits = model(inputs)
            losses = nn.functional.cross_entropy(
                logits,
                targets,
                reduction="none",
            )
            predictions = logits.argmax(dim=1)
            correct = predictions == targets
            total_loss += float(losses.sum())
            total_count += targets.numel()
            total_correct += int(correct.sum())

            for is_nine in (False, True):
                mask = (targets == 9) if is_nine else (targets != 9)
                group_loss[is_nine] += float(losses[mask].sum())
                group_count[is_nine] += int(mask.sum())
                group_correct[is_nine] += int(correct[mask].sum())

            cpu_targets = targets.cpu()
            cpu_correct = correct.cpu()
            class_count += torch.bincount(cpu_targets, minlength=10)
            class_correct += torch.bincount(
                cpu_targets[cpu_correct],
                minlength=10,
            )

    if was_training:
        model.train()
    if total_count == 0 or any(count == 0 for count in group_count.values()):
        raise ValueError("evaluation data must contain nine and non-nine observations")
    if bool((class_count == 0).any()):
        raise ValueError("evaluation data must contain every MNIST class")

    class_accuracy = class_correct.double() / class_count
    return {
        "sample_count": total_count,
        "nll": total_loss / total_count,
        "accuracy": total_correct / total_count,
        "non_nine_nll": group_loss[False] / group_count[False],
        "non_nine_accuracy": group_correct[False] / group_count[False],
        "nine_nll": group_loss[True] / group_count[True],
        "nine_accuracy": group_correct[True] / group_count[True],
        "balanced_accuracy": float(class_accuracy.mean()),
    }


def _build_optimizer(
    model: nn.Module,
    config: InitializationConfig,
) -> torch.optim.Optimizer:
    arguments = {
        "lr": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
    }
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **arguments)
    return torch.optim.SGD(model.parameters(), **arguments)


def fit_p0_initialization(
    model: CanonicalMnistCNN,
    train_dataset: Dataset,
    evaluation_dataset: Dataset,
    partitions: DatasetPartitions,
    config: InitializationConfig,
    *,
    loader_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> InitializationResult:
    """Fit the canonical model on the initialization partition containing no 9s."""

    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    initialization_targets = torch.as_tensor(
        getattr(train_dataset, "targets", []),
        dtype=torch.long,
    )[list(partitions.initialization)]
    if initialization_targets.numel() != len(partitions.initialization):
        raise ValueError("train dataset must expose targets for all observations")
    if bool((initialization_targets == 9).any()):
        raise ValueError("p=0 initialization data cannot contain digit 9")

    train_subset = Subset(train_dataset, partitions.initialization)
    evaluation_subset = Subset(evaluation_dataset, partitions.evaluation)
    loader = _make_loader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=loader_seed,
        num_workers=config.num_workers,
    )
    optimizer = _build_optimizer(model, config)
    history = []
    stopped_on_target = False

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=dtype)
            targets = targets.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * targets.numel()
            sample_count += targets.numel()

        metrics = evaluate_classifier(
            model,
            evaluation_subset,
            batch_size=config.batch_size,
            device=device,
            dtype=dtype,
            num_workers=config.num_workers,
        )
        history.append(
            {
                "epoch": epoch,
                "training_nll": total_loss / sample_count,
                **metrics,
            }
        )
        target = config.target_non_nine_accuracy
        if target is not None and metrics["non_nine_accuracy"] >= target:
            stopped_on_target = True
            break

    return InitializationResult(
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        wall_time_seconds=time.perf_counter() - start_time,
        epochs_completed=len(history),
        stopped_on_target=stopped_on_target,
        history=tuple(history),
        final_metrics=dict(history[-1]),
    )


def _state_dict_cpu(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in model.state_dict().items()
    }


def state_dict_hash(state_dict: dict[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in state_dict.items():
        cpu_tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu_tensor.dtype).encode("ascii"))
        digest.update(str(tuple(cpu_tensor.shape)).encode("ascii"))
        digest.update(cpu_tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_torch(path: Path, value: Any) -> None:
    with path.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def save_replica_bundle(
    root: str | Path,
    config: ExperimentConfig,
    model: CanonicalMnistCNN,
    layout: ParameterLayout,
    partitions: DatasetPartitions,
    stream_plan: MixtureStreamPlan,
    initialization: InitializationResult,
    *,
    device: torch.device,
    repo_root: str | Path,
) -> Path:
    """Atomically write an immutable shared initialization and stream bundle."""

    config.validate()
    layout.validate_module(model)
    partitions.validate()
    stream_plan.validate()
    if stream_plan.partition_hash != partitions.content_hash:
        raise ReplicaBundleError("stream plan does not belong to these partitions")

    bundle_id = replica_bundle_id(config)
    root_path = Path(root)
    destination = root_path / bundle_id
    if destination.exists():
        raise ReplicaBundleError(f"replica bundle already exists: {destination}")
    incomplete_root = root_path / ".incomplete"
    incomplete_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f"{bundle_id}.", dir=incomplete_root)
    )

    try:
        state_dict = _state_dict_cpu(model)
        model_hash = state_dict_hash(state_dict)
        metadata = {
            "bundle_schema_version": REPLICA_BUNDLE_SCHEMA_VERSION,
            "bundle_id": bundle_id,
            "replica_design_hash": replica_design_hash(config),
            "replica_design": replica_design_mapping(config),
            "model_state_hash": model_hash,
            "parameter_layout": layout.metadata(),
            "partition_hash": partitions.content_hash,
            "stream_plan_hash": stream_plan.content_hash,
            "execution_device": str(device),
            "training_dtype": config.runtime.training_dtype,
            "artifact_schema_version": config.artifact_schema_version,
            "metric_schema_version": config.metric_schema_version,
            "started_at": initialization.started_at,
            "completed_at": initialization.completed_at,
            "runtime": collect_runtime_metadata(repo_root, config),
            "status": "completed",
        }
        _write_torch(temporary / "model.pt", state_dict)
        _write_json(temporary / "partitions.json", partitions.to_mapping())
        _write_json(temporary / "stream_plan.json", stream_plan.to_mapping())
        _write_json(temporary / "initialization_metrics.json", initialization.to_mapping())
        _write_json(temporary / "metadata.json", metadata)
        with (temporary / "COMPLETED").open("wb") as marker:
            marker.flush()
            os.fsync(marker.fileno())
        root_path.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplicaBundleError(f"could not read {path}: {exc}") from exc


def load_replica_bundle(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedReplicaBundle:
    bundle_path = Path(path)
    if not (bundle_path / "COMPLETED").is_file():
        raise ReplicaBundleError(f"replica bundle is incomplete: {bundle_path}")
    metadata = _read_json(bundle_path / "metadata.json")
    if metadata.get("bundle_schema_version") != REPLICA_BUNDLE_SCHEMA_VERSION:
        raise ReplicaBundleError("unsupported replica bundle schema")

    dtype = resolve_dtype(metadata["training_dtype"])
    model, layout = build_canonical_model(0, device=device, dtype=dtype)
    try:
        state_dict = torch.load(
            bundle_path / "model.pt",
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError) as exc:
        raise ReplicaBundleError(f"could not load model state: {exc}") from exc
    if state_dict_hash(state_dict) != metadata.get("model_state_hash"):
        raise ReplicaBundleError("model state hash does not match metadata")
    try:
        layout.assert_metadata(metadata["parameter_layout"])
        model.load_state_dict(state_dict, strict=True)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise ReplicaBundleError(f"incompatible model artifact: {exc}") from exc
    model.to(device)

    partitions = DatasetPartitions.from_mapping(
        _read_json(bundle_path / "partitions.json")
    )
    stream_plan = MixtureStreamPlan.from_mapping(
        _read_json(bundle_path / "stream_plan.json")
    )
    if partitions.content_hash != metadata.get("partition_hash"):
        raise ReplicaBundleError("partition hash does not match metadata")
    if stream_plan.content_hash != metadata.get("stream_plan_hash"):
        raise ReplicaBundleError("stream plan hash does not match metadata")
    initialization_value = _read_json(
        bundle_path / "initialization_metrics.json"
    )
    initialization = InitializationResult(
        started_at=initialization_value["started_at"],
        completed_at=initialization_value["completed_at"],
        wall_time_seconds=initialization_value["wall_time_seconds"],
        epochs_completed=initialization_value["epochs_completed"],
        stopped_on_target=initialization_value["stopped_on_target"],
        history=tuple(initialization_value["history"]),
        final_metrics=initialization_value["final_metrics"],
    )
    return LoadedReplicaBundle(
        model=model,
        layout=layout,
        partitions=partitions,
        stream_plan=stream_plan,
        initialization=initialization,
        metadata=metadata,
    )


def load_replica_bundle_for_config(
    root: str | Path,
    config: ExperimentConfig,
    *,
    device: torch.device | str = "cpu",
) -> LoadedReplicaBundle:
    """Resolve and validate the shared bundle for any paired treatment config."""

    expected_id = replica_bundle_id(config)
    loaded = load_replica_bundle(Path(root) / expected_id, device=device)
    if loaded.metadata.get("bundle_id") != expected_id:
        raise ReplicaBundleError("replica bundle ID does not match configuration")
    if loaded.metadata.get("replica_design_hash") != replica_design_hash(config):
        raise ReplicaBundleError(
            "replica bundle design does not match configuration"
        )
    return loaded
