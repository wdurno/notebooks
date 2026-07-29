"""Reproducible MNIST partitions and paired mixture-stream plans."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import DataConfig
from .seeding import derive_seed_map

MNIST_DATA_SCHEMA_VERSION = 1
STREAM_PLAN_SCHEMA_VERSION = 1
REFERENCE_SAMPLE_PLAN_SCHEMA_VERSION = 2


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_long_targets(targets: Tensor | Sequence[int]) -> Tensor:
    tensor = torch.as_tensor(targets, dtype=torch.long, device="cpu")
    if tensor.ndim != 1 or tensor.numel() == 0:
        raise ValueError("targets must be a nonempty vector")
    if bool(((tensor < 0) | (tensor > 9)).any()):
        raise ValueError("MNIST targets must be integers from 0 through 9")
    return tensor


def _sample_without_replacement(
    candidates: Tensor,
    count: int,
    generator: torch.Generator,
) -> Tensor:
    if count > candidates.numel():
        raise ValueError(
            f"requested {count} observations from a pool of {candidates.numel()}"
        )
    order = torch.randperm(candidates.numel(), generator=generator)
    return candidates[order[:count]]


def _require_labels(
    targets: Tensor,
    indices: Tensor,
    labels: set[int],
    name: str,
) -> None:
    present = set(targets[indices].tolist())
    missing = sorted(labels - present)
    if missing:
        raise ValueError(
            f"{name} is missing MNIST labels {missing}; increase its configured size"
        )


@dataclasses.dataclass(frozen=True)
class DatasetPartitions:
    initialization: tuple[int, ...]
    online: tuple[int, ...]
    reference: tuple[int, ...]
    evaluation: tuple[int, ...]
    train_size: int
    test_size: int
    seed: int
    initialization_seed: int
    evaluation_seed: int
    schema_version: int = MNIST_DATA_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def content_hash(self) -> str:
        return _canonical_hash(self.to_mapping())

    def validate(self) -> None:
        if self.schema_version != MNIST_DATA_SCHEMA_VERSION:
            raise ValueError("unsupported MNIST partition schema")
        train_sets = (
            set(self.initialization),
            set(self.online),
            set(self.reference),
        )
        if any(
            index < 0 or index >= self.train_size
            for indices in train_sets
            for index in indices
        ):
            raise ValueError("train partition index is out of bounds")
        if any(
            index < 0 or index >= self.test_size
            for index in self.evaluation
        ):
            raise ValueError("evaluation partition index is out of bounds")
        if (
            train_sets[0] & train_sets[1]
            or train_sets[0] & train_sets[2]
            or train_sets[1] & train_sets[2]
        ):
            raise ValueError("train partitions must be pairwise disjoint")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "DatasetPartitions":
        converted = dict(value)
        for name in ("initialization", "online", "reference", "evaluation"):
            converted[name] = tuple(converted[name])
        partitions = cls(**converted)
        partitions.validate()
        return partitions


def partition_mnist(
    train_targets: Tensor | Sequence[int],
    test_targets: Tensor | Sequence[int],
    config: DataConfig,
    *,
    replica_seed: int,
) -> DatasetPartitions:
    """Create disjoint train pools and an independently sampled test subset."""

    config.validate()
    train = _as_long_targets(train_targets)
    test = _as_long_targets(test_targets)
    seeds = derive_seed_map(replica_seed)
    initialization_generator = torch.Generator().manual_seed(
        seeds["initialization_data"]
    )
    partition_generator = torch.Generator().manual_seed(seeds["data_partition"])
    evaluation_generator = torch.Generator().manual_seed(seeds["evaluation_data"])

    non_nine = torch.nonzero(train != 9, as_tuple=False).flatten()
    initialization = _sample_without_replacement(
        non_nine,
        config.initialization_size,
        initialization_generator,
    )

    available = torch.ones(train.numel(), dtype=torch.bool)
    available[initialization] = False
    remaining = torch.nonzero(available, as_tuple=False).flatten()
    requested_pool_size = config.online_pool_size + config.reference_pool_size
    selected = _sample_without_replacement(
        remaining,
        requested_pool_size,
        partition_generator,
    )
    online = selected[: config.online_pool_size]
    reference = selected[config.online_pool_size :]
    evaluation = _sample_without_replacement(
        torch.arange(test.numel()),
        config.evaluation_size,
        evaluation_generator,
    )

    _require_labels(train, initialization, set(range(9)), "initialization partition")
    _require_labels(train, online, set(range(10)), "online partition")
    _require_labels(train, reference, set(range(10)), "reference partition")
    _require_labels(test, evaluation, set(range(10)), "evaluation partition")

    partitions = DatasetPartitions(
        initialization=tuple(initialization.tolist()),
        online=tuple(online.tolist()),
        reference=tuple(reference.tolist()),
        evaluation=tuple(evaluation.tolist()),
        train_size=train.numel(),
        test_size=test.numel(),
        seed=seeds["data_partition"],
        initialization_seed=seeds["initialization_data"],
        evaluation_seed=seeds["evaluation_data"],
    )
    partitions.validate()
    return partitions


@dataclasses.dataclass(frozen=True)
class MixtureStreamPlan:
    p_values: tuple[float, ...]
    observation_indices: tuple[tuple[int, ...], ...]
    class_labels: tuple[tuple[int, ...], ...]
    samples_per_step: int
    non_nine_sampling: str
    seed: int
    partition_hash: str
    schema_version: int = STREAM_PLAN_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def content_hash(self) -> str:
        return _canonical_hash(self.to_mapping())

    def validate(self) -> None:
        if self.schema_version != STREAM_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported mixture stream schema")
        if len(self.p_values) < 2:
            raise ValueError("mixture stream requires at least two p values")
        if self.p_values[0] != 0.0 or self.p_values[-1] != 1.0:
            raise ValueError("mixture stream must include p=0 and p=1")
        expected_shape = (len(self.p_values), self.samples_per_step)
        if (
            len(self.observation_indices) != expected_shape[0]
            or len(self.class_labels) != expected_shape[0]
            or any(
                len(row) != expected_shape[1]
                for row in self.observation_indices
            )
            or any(len(row) != expected_shape[1] for row in self.class_labels)
        ):
            raise ValueError("stream arrays do not match the configured shape")
        if any(
            label < 0 or label > 9
            for row in self.class_labels
            for label in row
        ):
            raise ValueError("stream class labels must be from 0 through 9")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "MixtureStreamPlan":
        converted = dict(value)
        converted["p_values"] = tuple(converted["p_values"])
        for name in ("observation_indices", "class_labels"):
            converted[name] = tuple(tuple(row) for row in converted[name])
        plan = cls(**converted)
        plan.validate()
        return plan


def generate_mixture_stream(
    train_targets: Tensor | Sequence[int],
    partitions: DatasetPartitions,
    config: DataConfig,
    *,
    seed: int,
) -> MixtureStreamPlan:
    """Materialize class choices and source observations for a complete p path."""

    config.validate()
    partitions.validate()
    targets = _as_long_targets(train_targets)
    if targets.numel() != partitions.train_size:
        raise ValueError("train targets do not match partition metadata")

    generator = torch.Generator().manual_seed(seed)
    p_values = tuple(
        step / (config.num_p_steps - 1)
        for step in range(config.num_p_steps)
    )
    observation_rows = []
    label_rows = []
    for p_value in p_values:
        observations = _draw_mixture_observations(
            targets,
            torch.tensor(partitions.online, dtype=torch.long),
            p=p_value,
            sample_size=config.samples_per_step,
            non_nine_sampling=config.non_nine_sampling,
            generator=generator,
        )
        labels = targets[observations]
        observation_rows.append(tuple(observations.tolist()))
        label_rows.append(tuple(labels.tolist()))

    plan = MixtureStreamPlan(
        p_values=p_values,
        observation_indices=tuple(observation_rows),
        class_labels=tuple(label_rows),
        samples_per_step=config.samples_per_step,
        non_nine_sampling=config.non_nine_sampling,
        seed=seed,
        partition_hash=partitions.content_hash,
    )
    plan.validate()
    return plan


def _draw_mixture_observations(
    targets: Tensor,
    pool: Tensor,
    *,
    p: float,
    sample_size: int,
    non_nine_sampling: str,
    generator: torch.Generator,
) -> Tensor:
    pool_targets = targets[pool]
    nine_candidates = pool[pool_targets == 9]
    non_nine_candidates = pool[pool_targets != 9]
    per_class_candidates = {
        label: pool[pool_targets == label] for label in range(9)
    }
    if nine_candidates.numel() == 0 or non_nine_candidates.numel() == 0:
        raise ValueError("sampling pool must contain nine and non-nine observations")
    if non_nine_sampling == "balanced" and any(
        candidates.numel() == 0 for candidates in per_class_candidates.values()
    ):
        raise ValueError("balanced sampling requires every non-nine class")

    is_nine = torch.rand(sample_size, generator=generator) < p
    observations = torch.empty(sample_size, dtype=torch.long)
    nine_count = int(is_nine.sum())
    if nine_count:
        choices = torch.randint(
            nine_candidates.numel(),
            (nine_count,),
            generator=generator,
        )
        observations[is_nine] = nine_candidates[choices]

    non_nine_count = sample_size - nine_count
    if non_nine_count:
        if non_nine_sampling == "empirical":
            choices = torch.randint(
                non_nine_candidates.numel(),
                (non_nine_count,),
                generator=generator,
            )
            observations[~is_nine] = non_nine_candidates[choices]
        else:
            labels = torch.randint(9, (non_nine_count,), generator=generator)
            selected = []
            for label in labels.tolist():
                candidates = per_class_candidates[label]
                choice = torch.randint(
                    candidates.numel(),
                    (),
                    generator=generator,
                )
                selected.append(int(candidates[choice]))
            observations[~is_nine] = torch.tensor(selected)
    return observations


@dataclasses.dataclass(frozen=True)
class ReferenceSamplePlan:
    p: float
    observation_indices: tuple[int, ...]
    class_labels: tuple[int, ...]
    non_nine_sampling: str
    seed: int
    partition_hash: str
    pool: str = "reference"
    schema_version: int = REFERENCE_SAMPLE_PLAN_SCHEMA_VERSION

    @property
    def sample_size(self) -> int:
        return len(self.observation_indices)

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def content_hash(self) -> str:
        return _canonical_hash(self.to_mapping())

    def validate(self) -> None:
        if self.schema_version != REFERENCE_SAMPLE_PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported reference sample-plan schema")
        if self.pool not in {"online", "reference"}:
            raise ValueError("reference sample pool must be 'online' or 'reference'")
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("reference p must be in [0, 1]")
        if not self.observation_indices:
            raise ValueError("reference sample plan cannot be empty")
        if len(self.observation_indices) != len(self.class_labels):
            raise ValueError("reference observation and label counts differ")
        if any(label < 0 or label > 9 for label in self.class_labels):
            raise ValueError("reference labels must be from 0 through 9")

    def prefix(self, sample_size: int) -> "ReferenceSamplePlan":
        if not 1 <= sample_size <= self.sample_size:
            raise ValueError("prefix sample size is out of bounds")
        plan = dataclasses.replace(
            self,
            observation_indices=self.observation_indices[:sample_size],
            class_labels=self.class_labels[:sample_size],
        )
        plan.validate()
        return plan

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ReferenceSamplePlan":
        converted = dict(value)
        converted["observation_indices"] = tuple(
            converted["observation_indices"]
        )
        converted["class_labels"] = tuple(converted["class_labels"])
        plan = cls(**converted)
        plan.validate()
        return plan


def generate_reference_sample_plan(
    train_targets: Tensor | Sequence[int],
    partitions: DatasetPartitions,
    config: DataConfig,
    *,
    p: float,
    sample_size: int,
    seed: int,
    pool: str = "reference",
) -> ReferenceSamplePlan:
    """Draw one nested, independently seeded reference sequence."""

    config.validate()
    partitions.validate()
    if not isinstance(sample_size, int) or isinstance(sample_size, bool):
        raise ValueError("sample_size must be an integer")
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if pool not in {"online", "reference"}:
        raise ValueError("pool must be 'online' or 'reference'")

    targets = _as_long_targets(train_targets)
    if targets.numel() != partitions.train_size:
        raise ValueError("train targets do not match partition metadata")
    observations = _draw_nested_mixture_observations(
        targets,
        torch.tensor(getattr(partitions, pool), dtype=torch.long),
        p=p,
        sample_size=sample_size,
        non_nine_sampling=config.non_nine_sampling,
        seed=seed,
    )
    plan = ReferenceSamplePlan(
        p=float(p),
        observation_indices=tuple(observations.tolist()),
        class_labels=tuple(targets[observations].tolist()),
        non_nine_sampling=config.non_nine_sampling,
        seed=seed,
        partition_hash=partitions.content_hash,
        pool=pool,
    )
    plan.validate()
    return plan


def _draw_nested_mixture_observations(
    targets: Tensor,
    pool: Tensor,
    *,
    p: float,
    sample_size: int,
    non_nine_sampling: str,
    seed: int,
) -> Tensor:
    """Draw a sequence whose prefixes do not depend on the requested maximum."""

    pool_targets = targets[pool]
    nine_candidates = pool[pool_targets == 9]
    non_nine_candidates = pool[pool_targets != 9]
    per_class_candidates = {
        label: pool[pool_targets == label] for label in range(9)
    }
    if nine_candidates.numel() == 0 or non_nine_candidates.numel() == 0:
        raise ValueError("sampling pool must contain nine and non-nine observations")
    if non_nine_sampling == "balanced" and any(
        candidates.numel() == 0 for candidates in per_class_candidates.values()
    ):
        raise ValueError("balanced sampling requires every non-nine class")

    mixture_generator = torch.Generator().manual_seed(seed)
    choice_generator = torch.Generator().manual_seed(
        (seed + 0x1F123BB5) % 2**63
    )
    class_generator = torch.Generator().manual_seed(
        (seed + 0x5F356495) % 2**63
    )
    is_nine = torch.rand(sample_size, generator=mixture_generator) < p
    choices = torch.rand(sample_size, generator=choice_generator)
    class_choices = torch.randint(
        9,
        (sample_size,),
        generator=class_generator,
    )
    observations = torch.empty(sample_size, dtype=torch.long)

    for index in range(sample_size):
        if bool(is_nine[index]):
            candidate_index = min(
                int(choices[index] * nine_candidates.numel()),
                nine_candidates.numel() - 1,
            )
            observations[index] = nine_candidates[candidate_index]
        elif non_nine_sampling == "empirical":
            candidate_index = min(
                int(choices[index] * non_nine_candidates.numel()),
                non_nine_candidates.numel() - 1,
            )
            observations[index] = non_nine_candidates[candidate_index]
        else:
            candidates = per_class_candidates[int(class_choices[index])]
            candidate_index = min(
                int(choices[index] * candidates.numel()),
                candidates.numel() - 1,
            )
            observations[index] = candidates[candidate_index]
    return observations


def load_mnist_datasets(
    root: str | Path,
    *,
    download: bool,
) -> tuple[Dataset, Dataset]:
    """Load canonical unaugmented MNIST train and test datasets."""

    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor

    dataset_root = Path(root)
    train = MNIST(dataset_root, train=True, transform=ToTensor(), download=download)
    test = MNIST(dataset_root, train=False, transform=ToTensor(), download=download)
    return train, test


def dataset_targets(dataset: Dataset) -> Tensor:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("dataset must expose a targets attribute")
    return _as_long_targets(targets)
