"""All-digit partitions and paired rotated streams for Plan 5."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.seeding import derive_component_seed

from .config import DataConfig, RotationConfig
from .schedule import RotationSchedule
from .transform import rotate_mnist_tensor, tensor_content_hash


PARTITION_SCHEMA_VERSION = 1
STREAM_SCHEMA_VERSION = 1


def _hash(value: Any) -> str:
    payload = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _targets(value: Tensor | Sequence[int]) -> Tensor:
    result = torch.as_tensor(value, dtype=torch.long, device="cpu")
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError("targets must be a nonempty vector")
    if bool(((result < 0) | (result > 9)).any()):
        raise ValueError("targets must contain MNIST labels 0 through 9")
    return result


def _require_all_classes(targets: Tensor, indices: tuple[int, ...], name: str) -> None:
    present = set(targets[list(indices)].tolist())
    missing = sorted(set(range(10)) - present)
    if missing:
        raise ValueError(f"{name} is missing MNIST labels {missing}")


@dataclasses.dataclass(frozen=True)
class RotatedPartitions:
    initialization: tuple[int, ...]
    online: tuple[int, ...]
    reference: tuple[int, ...]
    evaluation: tuple[int, ...]
    train_size: int
    test_size: int
    partition_seed: int
    evaluation_seed: int
    schema_version: int = PARTITION_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def content_hash(self) -> str:
        return _hash(self.to_mapping())

    def validate(self) -> None:
        if self.schema_version != PARTITION_SCHEMA_VERSION:
            raise ValueError("unsupported rotated partition schema")
        train_sets = tuple(
            set(values)
            for values in (self.initialization, self.online, self.reference)
        )
        if any(
            index < 0 or index >= self.train_size
            for values in train_sets
            for index in values
        ):
            raise ValueError("rotated train partition index is out of bounds")
        if any(index < 0 or index >= self.test_size for index in self.evaluation):
            raise ValueError("rotated evaluation index is out of bounds")
        if train_sets[0] & train_sets[1] or train_sets[0] & train_sets[2] or train_sets[1] & train_sets[2]:
            raise ValueError("rotated train partitions must be disjoint")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RotatedPartitions":
        converted = dict(value)
        for name in ("initialization", "online", "reference", "evaluation"):
            converted[name] = tuple(converted[name])
        result = cls(**converted)
        result.validate()
        return result


def partition_all_digit_mnist(
    train_targets: Tensor | Sequence[int],
    test_targets: Tensor | Sequence[int],
    config: DataConfig,
    *,
    replica_seed: int,
) -> RotatedPartitions:
    train = _targets(train_targets)
    test = _targets(test_targets)
    requested = (
        config.initialization_size
        + config.online_pool_size
        + config.reference_pool_size
    )
    if requested > train.numel() or config.evaluation_size > test.numel():
        raise ValueError("rotated partition sizes exceed the MNIST datasets")
    partition_seed = derive_component_seed(replica_seed, "plan5_data_partition")
    evaluation_seed = derive_component_seed(replica_seed, "plan5_evaluation_panel")
    train_order = torch.randperm(
        train.numel(), generator=torch.Generator().manual_seed(partition_seed)
    )[:requested]
    evaluation = torch.randperm(
        test.numel(), generator=torch.Generator().manual_seed(evaluation_seed)
    )[: config.evaluation_size]
    left = config.initialization_size
    right = left + config.online_pool_size
    result = RotatedPartitions(
        initialization=tuple(train_order[:left].tolist()),
        online=tuple(train_order[left:right].tolist()),
        reference=tuple(train_order[right:].tolist()),
        evaluation=tuple(evaluation.tolist()),
        train_size=train.numel(),
        test_size=test.numel(),
        partition_seed=partition_seed,
        evaluation_seed=evaluation_seed,
    )
    result.validate()
    _require_all_classes(train, result.initialization, "initialization partition")
    _require_all_classes(train, result.online, "online partition")
    _require_all_classes(train, result.reference, "reference partition")
    _require_all_classes(test, result.evaluation, "evaluation partition")
    return result


@dataclasses.dataclass(frozen=True)
class RotatedStreamPlan:
    schedule: RotationSchedule
    observation_indices: tuple[tuple[int, ...], ...]
    class_labels: tuple[tuple[int, ...], ...]
    transformed_hashes: tuple[tuple[str, ...], ...]
    samples_per_step: int
    master_samples_per_step: int
    seed: int
    partition_hash: str
    tensor_artifact_hash: str
    schema_version: int = STREAM_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schedule": self.schedule.to_mapping(),
            "observation_indices": [list(row) for row in self.observation_indices],
            "class_labels": [list(row) for row in self.class_labels],
            "transformed_hashes": [list(row) for row in self.transformed_hashes],
            "samples_per_step": self.samples_per_step,
            "master_samples_per_step": self.master_samples_per_step,
            "seed": self.seed,
            "partition_hash": self.partition_hash,
            "tensor_artifact_hash": self.tensor_artifact_hash,
            "schema_version": self.schema_version,
        }

    @property
    def content_hash(self) -> str:
        return _hash(self.to_mapping())

    def validate(self) -> None:
        self.schedule.validate()
        if self.schema_version != STREAM_SCHEMA_VERSION:
            raise ValueError("unsupported rotated stream schema")
        expected_rows = self.schedule.num_points
        if any(
            len(values) != expected_rows
            for values in (
                self.observation_indices,
                self.class_labels,
                self.transformed_hashes,
            )
        ):
            raise ValueError("rotated stream row count does not match schedule")
        if any(
            len(row) != self.samples_per_step
            for values in (
                self.observation_indices,
                self.class_labels,
                self.transformed_hashes,
            )
            for row in values
        ):
            raise ValueError("rotated stream batch width is inconsistent")
        if self.master_samples_per_step < self.samples_per_step:
            raise ValueError("rotated stream master width is smaller than its batch")
        if any(
            label < 0 or label > 9 for row in self.class_labels for label in row
        ):
            raise ValueError("rotated stream labels must be MNIST classes")
        for value in (self.partition_hash, self.tensor_artifact_hash):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("rotated stream contains an invalid hash")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RotatedStreamPlan":
        expected = {
            "schedule",
            "observation_indices",
            "class_labels",
            "transformed_hashes",
            "samples_per_step",
            "master_samples_per_step",
            "seed",
            "partition_hash",
            "tensor_artifact_hash",
            "schema_version",
        }
        if set(value) != expected:
            raise ValueError("rotated stream mapping has invalid fields")
        result = cls(
            schedule=RotationSchedule.from_mapping(value["schedule"]),
            observation_indices=tuple(
                tuple(row) for row in value["observation_indices"]
            ),
            class_labels=tuple(tuple(row) for row in value["class_labels"]),
            transformed_hashes=tuple(
                tuple(row) for row in value["transformed_hashes"]
            ),
            samples_per_step=value["samples_per_step"],
            master_samples_per_step=value["master_samples_per_step"],
            seed=value["seed"],
            partition_hash=value["partition_hash"],
            tensor_artifact_hash=value["tensor_artifact_hash"],
            schema_version=value["schema_version"],
        )
        result.validate()
        return result


def generate_rotated_stream(
    dataset: Dataset,
    targets: Tensor | Sequence[int],
    partitions: RotatedPartitions,
    schedule: RotationSchedule,
    data_config: DataConfig,
    rotation_config: RotationConfig,
    *,
    replica_seed: int,
) -> tuple[RotatedStreamPlan, Tensor, Tensor]:
    labels = _targets(targets)
    if labels.numel() != partitions.train_size:
        raise ValueError("dataset targets do not match rotated partitions")
    seed = derive_component_seed(replica_seed, "plan5_online_stream")
    generator = torch.Generator().manual_seed(seed)
    pool = torch.tensor(partitions.online, dtype=torch.long)
    choice_shape = (schedule.num_points, data_config.stream_width)
    master_choices = torch.randint(pool.numel(), choice_shape, generator=generator)
    choices = master_choices[:, : data_config.samples_per_step]
    indices = pool[choices]
    target_tensor = labels[indices]
    batches = []
    transformed_hashes = []
    for step, angle in enumerate(schedule.angles_degrees):
        images = []
        hashes = []
        for index in indices[step].tolist():
            image, target = dataset[index]
            if int(target) != int(labels[index]):
                raise RuntimeError("dataset target differs from target metadata")
            rotated = rotate_mnist_tensor(image, angle, rotation_config)
            images.append(rotated)
            hashes.append(tensor_content_hash(rotated))
        batches.append(torch.stack(images))
        transformed_hashes.append(tuple(hashes))
    inputs = torch.stack(batches).contiguous()
    targets_tensor = target_tensor.contiguous()
    artifact_hash = tensor_content_hash(inputs) + tensor_content_hash(targets_tensor)
    artifact_hash = hashlib.sha256(artifact_hash.encode("ascii")).hexdigest()
    plan = RotatedStreamPlan(
        schedule=schedule,
        observation_indices=tuple(tuple(row) for row in indices.tolist()),
        class_labels=tuple(tuple(row) for row in target_tensor.tolist()),
        transformed_hashes=tuple(transformed_hashes),
        samples_per_step=data_config.samples_per_step,
        master_samples_per_step=data_config.stream_width,
        seed=seed,
        partition_hash=partitions.content_hash,
        tensor_artifact_hash=artifact_hash,
    )
    plan.validate()
    return plan, inputs, targets_tensor


def validate_stream_tensors(
    plan: RotatedStreamPlan,
    inputs: Tensor,
    targets: Tensor,
) -> None:
    expected_inputs = (
        plan.schedule.num_points,
        plan.samples_per_step,
        1,
        28,
        28,
    )
    expected_targets = (plan.schedule.num_points, plan.samples_per_step)
    if tuple(inputs.shape) != expected_inputs or tuple(targets.shape) != expected_targets:
        raise ValueError("rotated stream tensor shape does not match its plan")
    if tuple(tuple(row) for row in targets.cpu().tolist()) != plan.class_labels:
        raise ValueError("rotated stream targets do not match its plan")
    combined = tensor_content_hash(inputs) + tensor_content_hash(targets)
    if hashlib.sha256(combined.encode("ascii")).hexdigest() != plan.tensor_artifact_hash:
        raise ValueError("rotated stream tensor artifact hash does not match")


class RotatedDatasetView(Dataset):
    """Read-only fixed-angle view over selected canonical observations."""

    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        *,
        angle_degrees: float,
        rotation_config: RotationConfig,
    ) -> None:
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        if not self.indices:
            raise ValueError("rotated dataset view requires observations")
        self.angle_degrees = float(angle_degrees)
        self.rotation_config = rotation_config

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, int]:
        image, target = self.dataset[self.indices[item]]
        return (
            rotate_mnist_tensor(
                image,
                self.angle_degrees,
                self.rotation_config,
            ),
            int(target),
        )
