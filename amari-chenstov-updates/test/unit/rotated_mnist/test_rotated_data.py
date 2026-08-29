import dataclasses
from pathlib import Path

import torch
from torch.utils.data import Dataset

from mnist_experiment.rotated_mnist.config import load_config
from mnist_experiment.rotated_mnist.data import (
    generate_rotated_stream,
    partition_all_digit_mnist,
    validate_stream_tensors,
)
from mnist_experiment.rotated_mnist.schedule import resolve_rotation_schedule


SMOKE_CONFIG = (
    Path(__file__).parents[3]
    / "mnist_experiment"
    / "rotated_mnist"
    / "configs"
    / "phase1_smoke.json"
)


class _TensorMnist(Dataset):
    def __init__(self, count_per_class: int) -> None:
        self.targets = torch.arange(10).repeat_interleave(count_per_class)
        self.images = torch.zeros(self.targets.numel(), 1, 28, 28)
        for index in range(self.targets.numel()):
            self.images[index, 0, 5:23, 5:23] = (index + 1) / self.targets.numel()

    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, index: int):
        return self.images[index], int(self.targets[index])


def test_streams_use_nested_master_width_prefixes() -> None:
    config = load_config(SMOKE_CONFIG)
    train = _TensorMnist(40)
    test = _TensorMnist(20)
    data = dataclasses.replace(
        config.data,
        initialization_size=100,
        online_pool_size=200,
        reference_pool_size=100,
        evaluation_size=100,
    )
    partitions = partition_all_digit_mnist(
        train.targets,
        test.targets,
        data,
        replica_seed=config.replica_seed,
    )
    schedule = resolve_rotation_schedule(config.rotation)
    plan2, inputs2, targets2 = generate_rotated_stream(
        train,
        train.targets,
        partitions,
        schedule,
        data,
        config.rotation,
        replica_seed=config.replica_seed,
    )
    wider = dataclasses.replace(data, samples_per_step=4)
    plan4, inputs4, targets4 = generate_rotated_stream(
        train,
        train.targets,
        partitions,
        schedule,
        wider,
        config.rotation,
        replica_seed=config.replica_seed,
    )

    assert plan2.observation_indices == tuple(
        row[:2] for row in plan4.observation_indices
    )
    assert torch.equal(inputs2, inputs4[:, :2])
    assert torch.equal(targets2, targets4[:, :2])
    validate_stream_tensors(plan2, inputs2, targets2)
    assert plan2.partition_hash == partitions.content_hash
