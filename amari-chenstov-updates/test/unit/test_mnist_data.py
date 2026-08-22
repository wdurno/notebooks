import dataclasses

import pytest
import torch

from src.config import DataConfig, ScheduleConfig
from src.mnist_data import (
    DatasetPartitions,
    MixtureStreamPlan,
    ReferenceSamplePlan,
    generate_mixture_stream,
    generate_reference_sample_plan,
    partition_mnist,
)
from src.seeding import derive_seed_map


def _data_config(
    *,
    non_nine_sampling: str = "empirical",
    num_p_steps: int = 5,
    samples_per_step: int = 5000,
    schedule: ScheduleConfig | None = None,
) -> DataConfig:
    return DataConfig(
        num_p_steps=num_p_steps,
        samples_per_step=samples_per_step,
        non_nine_sampling=non_nine_sampling,
        initialization_size=180,
        online_pool_size=500,
        reference_pool_size=500,
        evaluation_size=200,
        schedule=schedule,
    )


def _targets() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.arange(2000) % 10, torch.arange(400) % 10


def test_mnist_partitions_are_deterministic_and_disjoint() -> None:
    train_targets, test_targets = _targets()
    config = _data_config()

    first = partition_mnist(
        train_targets,
        test_targets,
        config,
        replica_seed=1729,
    )
    second = partition_mnist(
        train_targets,
        test_targets,
        config,
        replica_seed=1729,
    )

    assert first == second
    assert len(first.initialization) == config.initialization_size
    assert len(first.online) == config.online_pool_size
    assert len(first.reference) == config.reference_pool_size
    assert len(first.evaluation) == config.evaluation_size
    assert not (
        set(first.initialization) & set(first.online)
        or set(first.initialization) & set(first.reference)
        or set(first.online) & set(first.reference)
    )
    assert not bool((train_targets[list(first.initialization)] == 9).any())


def test_mixture_stream_is_deterministic_and_materializes_valid_observations() -> None:
    train_targets, test_targets = _targets()
    config = _data_config(samples_per_step=100)
    partitions = partition_mnist(
        train_targets,
        test_targets,
        config,
        replica_seed=1729,
    )
    seed = derive_seed_map(1729)["online_stream"]

    first = generate_mixture_stream(
        train_targets,
        partitions,
        config,
        seed=seed,
    )
    second = generate_mixture_stream(
        train_targets,
        partitions,
        config,
        seed=seed,
    )
    restored = MixtureStreamPlan.from_mapping(first.to_mapping())

    assert first == second == restored
    assert first.content_hash == restored.content_hash
    assert set(first.observation_indices[0]).issubset(set(partitions.online))
    assert all(label != 9 for label in first.class_labels[0])
    assert all(label == 9 for label in first.class_labels[-1])
    for indices, labels in zip(
        first.observation_indices,
        first.class_labels,
        strict=True,
    ):
        assert train_targets[list(indices)].tolist() == list(labels)


def test_empirical_stream_tracks_p_and_fixed_non_nine_distribution() -> None:
    train_targets, test_targets = _targets()
    config = _data_config()
    partitions = partition_mnist(
        train_targets,
        test_targets,
        config,
        replica_seed=42,
    )
    plan = generate_mixture_stream(
        train_targets,
        partitions,
        config,
        seed=derive_seed_map(42)["online_stream"],
    )
    online_targets = train_targets[list(partitions.online)]
    expected = torch.bincount(
        online_targets[online_targets != 9],
        minlength=9,
    ).double()
    expected /= expected.sum()

    for p_value, labels in zip(plan.p_values[1:-1], plan.class_labels[1:-1]):
        label_tensor = torch.tensor(labels)
        observed_p = float((label_tensor == 9).double().mean())
        non_nine = label_tensor[label_tensor != 9]
        observed = torch.bincount(non_nine, minlength=9).double()
        observed /= observed.sum()

        assert abs(observed_p - p_value) < 0.03
        torch.testing.assert_close(observed, expected, atol=0.035, rtol=0)


def test_balanced_non_nine_sampling_is_uniform() -> None:
    train_targets, test_targets = _targets()
    config = _data_config(
        non_nine_sampling="balanced",
        num_p_steps=3,
        samples_per_step=9000,
    )
    partitions = partition_mnist(
        train_targets,
        test_targets,
        config,
        replica_seed=7,
    )
    plan = generate_mixture_stream(
        train_targets,
        partitions,
        config,
        seed=derive_seed_map(7)["online_stream"],
    )
    middle = torch.tensor(plan.class_labels[1])
    non_nine = middle[middle != 9]
    proportions = torch.bincount(non_nine, minlength=9).double()
    proportions /= proportions.sum()

    torch.testing.assert_close(
        proportions,
        torch.full((9,), 1 / 9, dtype=torch.float64),
        atol=0.02,
        rtol=0,
    )


def test_partition_serialization_preserves_content_hash() -> None:
    train_targets, test_targets = _targets()
    partitions = partition_mnist(
        train_targets,
        test_targets,
        _data_config(),
        replica_seed=9,
    )

    restored = DatasetPartitions.from_mapping(partitions.to_mapping())

    assert restored == partitions
    assert restored.content_hash == partitions.content_hash


def test_reference_sample_prefixes_are_nested_and_reproducible() -> None:
    train_targets, test_targets = _targets()
    config = _data_config()
    partitions = partition_mnist(
        train_targets,
        test_targets,
        config,
        replica_seed=12,
    )
    seed = derive_seed_map(12)["reference_stream"]
    full = generate_reference_sample_plan(
        train_targets,
        partitions,
        config,
        p=0.5,
        sample_size=100,
        seed=seed,
    )
    repeated = generate_reference_sample_plan(
        train_targets,
        partitions,
        config,
        p=0.5,
        sample_size=100,
        seed=seed,
    )
    prefix = full.prefix(40)
    independently_short = generate_reference_sample_plan(
        train_targets,
        partitions,
        config,
        p=0.5,
        sample_size=40,
        seed=seed,
    )
    restored = ReferenceSamplePlan.from_mapping(full.to_mapping())

    assert full == repeated == restored
    assert prefix.observation_indices == full.observation_indices[:40]
    assert prefix.class_labels == full.class_labels[:40]
    assert independently_short == prefix
    assert prefix.content_hash != full.content_hash


def test_mixture_stream_prefixes_each_ordered_batch_without_resampling() -> None:
    targets, test_targets = _targets()
    config = _data_config(samples_per_step=8)
    partitions = partition_mnist(
        targets,
        test_targets,
        config,
        replica_seed=44,
    )
    full = generate_mixture_stream(targets, partitions, config, seed=44)

    prefix = full.prefix_per_step(1)

    assert prefix.samples_per_step == 1
    assert prefix.p_values == full.p_values
    assert prefix.partition_hash == full.partition_hash
    assert prefix.seed == full.seed
    assert prefix.observation_indices == tuple(
        row[:1] for row in full.observation_indices
    )
    assert prefix.class_labels == tuple(row[:1] for row in full.class_labels)
    assert prefix.content_hash != full.content_hash
    with pytest.raises(ValueError, match="prefix samples_per_step"):
        full.prefix_per_step(9)


def test_explicit_schedules_share_common_uniform_and_candidate_streams() -> None:
    train_targets, test_targets = _targets()
    first_config = _data_config(
        num_p_steps=9,
        samples_per_step=100,
        schedule=ScheduleConfig(
            kind="normalized_logistic",
            p_start=0.0,
            p_end=0.2,
            center_fraction=0.5,
            steepness=8.0,
        ),
    )
    second_config = dataclasses.replace(
        first_config,
        schedule=dataclasses.replace(first_config.schedule, steepness=32.0),
    )
    partitions = partition_mnist(
        train_targets,
        test_targets,
        first_config,
        replica_seed=1729,
    )

    first = generate_mixture_stream(
        train_targets,
        partitions,
        first_config,
        seed=derive_seed_map(1729)["online_stream"],
    )
    second = generate_mixture_stream(
        train_targets,
        partitions,
        second_config,
        seed=derive_seed_map(1729)["online_stream"],
    )

    assert first.schema_version == second.schema_version == 2
    assert first.uniform_stream_hash == second.uniform_stream_hash
    assert first.schedule_hash != second.schedule_hash
    assert first.content_hash != second.content_hash
    assert first.observation_indices[0] == second.observation_indices[0]
    assert first.observation_indices[-1] == second.observation_indices[-1]
    assert first.class_labels[0] == second.class_labels[0]
    assert first.class_labels[-1] == second.class_labels[-1]
    assert MixtureStreamPlan.from_mapping(first.to_mapping()) == first


def test_scheduled_stream_accepts_recorded_zero_speed_tail_transitions() -> None:
    train_targets, test_targets = _targets()
    config = _data_config(
        num_p_steps=100,
        samples_per_step=1,
        schedule=ScheduleConfig(
            kind="normalized_logistic",
            p_start=0.0,
            p_end=0.2,
            center_fraction=0.5,
            steepness=256.0,
        ),
    )
    partitions = partition_mnist(
        train_targets,
        test_targets,
        config,
        replica_seed=1729,
    )

    plan = generate_mixture_stream(
        train_targets,
        partitions,
        config,
        seed=derive_seed_map(1729)["online_stream"],
    )

    assert any(
        right == left
        for left, right in zip(plan.p_values, plan.p_values[1:])
    )
    assert plan.p_values[0] == 0.0
    assert plan.p_values[-1] == 0.2
