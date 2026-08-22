import dataclasses
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from src.config import EstimatorConfig, ScheduleConfig, load_config
from src.initialization import (
    InitializationResult,
    ReplicaBundleError,
    derive_replica_bundle,
    derive_scheduled_replica_bundle,
    fit_p0_initialization,
    load_replica_bundle,
    load_replica_bundle_for_config,
    replica_bundle_id,
    replica_design_hash,
    save_replica_bundle,
)
from src.mnist_data import DatasetPartitions, generate_mixture_stream
from src.mnist_model import build_canonical_model
from src.seeding import derive_seed_map


REPO_ROOT = Path(__file__).parents[2]
SMOKE_CONFIG = REPO_ROOT / "mnist_experiment" / "configs" / "smoke.json"


class SyntheticMnist(Dataset):
    def __init__(self, targets: torch.Tensor) -> None:
        self.targets = targets.clone()
        self.images = torch.zeros(targets.numel(), 1, 28, 28)
        for index, label in enumerate(targets.tolist()):
            row = 2 * label
            self.images[index, 0, row : row + 2, :] = 1.0

    def __len__(self) -> int:
        return self.targets.numel()

    def __getitem__(self, index: int):
        return self.images[index], self.targets[index]


def _tiny_fixture():
    config = load_config(SMOKE_CONFIG)
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            initialization_size=9,
            online_pool_size=40,
            reference_pool_size=40,
            evaluation_size=20,
            samples_per_step=4,
        ),
        initialization=dataclasses.replace(
            config.initialization,
            batch_size=9,
            max_epochs=1,
        ),
        runtime=dataclasses.replace(
            config.runtime,
            training_dtype="float32",
        ),
    )
    train_targets = torch.arange(100) % 10
    test_targets = torch.arange(20) % 10
    train_dataset = SyntheticMnist(train_targets)
    test_dataset = SyntheticMnist(test_targets)
    partitions = DatasetPartitions(
        initialization=tuple(range(9)),
        online=tuple(range(10, 50)),
        reference=tuple(range(50, 90)),
        evaluation=tuple(range(20)),
        train_size=100,
        test_size=20,
        seed=1,
        initialization_seed=2,
        evaluation_seed=3,
    )
    seeds = derive_seed_map(config.replica_seed)
    stream = generate_mixture_stream(
        train_targets,
        partitions,
        config.data,
        seed=seeds["online_stream"],
    )
    model, layout = build_canonical_model(
        seeds["initialization"],
        dtype=torch.float32,
    )
    return config, train_dataset, test_dataset, partitions, stream, model, layout


def test_replica_design_identity_excludes_fisher_treatment() -> None:
    config = load_config(SMOKE_CONFIG)
    assert replica_design_hash(config) == (
        "7527ae76131c4c07bf52a14cdd1d181dbe9cbe108e97dd75d813df808d61736a"
    )
    other_estimator = EstimatorConfig(
        method="ema",
        representation="dense",
        ema_gain=config.estimator.ema_gain,
        fresh_fisher_cadence=None,
        low_rank=None,
    )
    paired = dataclasses.replace(config, estimator=other_estimator)

    assert replica_design_hash(config) == replica_design_hash(paired)
    assert replica_bundle_id(config) == replica_bundle_id(paired)
    assert config.config_hash != paired.config_hash

    cpu = dataclasses.replace(
        config,
        runtime=dataclasses.replace(config.runtime, device="cpu"),
    )
    cuda = dataclasses.replace(
        config,
        runtime=dataclasses.replace(config.runtime, device="cuda"),
    )
    assert replica_design_hash(cpu) == replica_design_hash(cuda)
    assert replica_bundle_id(cpu) == replica_bundle_id(cuda)


def test_tiny_initialization_bundle_round_trip_is_exact(tmp_path: Path) -> None:
    (
        config,
        train_dataset,
        test_dataset,
        partitions,
        stream,
        model,
        layout,
    ) = _tiny_fixture()
    seeds = derive_seed_map(config.replica_seed)
    result = fit_p0_initialization(
        model,
        train_dataset,
        test_dataset,
        partitions,
        config.initialization,
        loader_seed=seeds["initialization_loader"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_state = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }

    path = save_replica_bundle(
        tmp_path,
        config,
        model,
        layout,
        partitions,
        stream,
        result,
        device=torch.device("cpu"),
        repo_root=REPO_ROOT,
    )
    loaded = load_replica_bundle(path)
    paired_estimator = dataclasses.replace(
        config,
        estimator=EstimatorConfig(
            method="ema",
            representation="dense",
            ema_gain=config.estimator.ema_gain,
            fresh_fisher_cadence=None,
            low_rank=None,
        ),
    )
    paired_loaded = load_replica_bundle_for_config(
        tmp_path,
        paired_estimator,
    )

    assert (path / "COMPLETED").is_file()
    assert loaded.layout.metadata() == layout.metadata()
    assert loaded.partitions == partitions
    assert loaded.stream_plan == stream
    assert paired_loaded.stream_plan == stream
    assert loaded.initialization.final_metrics == result.final_metrics
    for name, tensor in loaded.model.state_dict().items():
        assert torch.equal(tensor, expected_state[name])
        assert torch.equal(
            paired_loaded.model.state_dict()[name],
            expected_state[name],
        )

    with pytest.raises(ReplicaBundleError, match="already exists"):
        save_replica_bundle(
            tmp_path,
            config,
            model,
            layout,
            partitions,
            stream,
            result,
            device=torch.device("cpu"),
            repo_root=REPO_ROOT,
        )


def test_derived_replica_bundle_is_an_exact_immutable_stream_prefix(
    tmp_path: Path,
) -> None:
    config, _, _, partitions, stream, model, layout = _tiny_fixture()
    initialization = InitializationResult(
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        wall_time_seconds=1.0,
        epochs_completed=1,
        stopped_on_target=False,
        history=(),
        final_metrics={"accuracy": 0.5},
    )
    parent_path = save_replica_bundle(
        tmp_path,
        config,
        model,
        layout,
        partitions,
        stream,
        initialization,
        device=torch.device("cpu"),
        repo_root=REPO_ROOT,
    )
    derived_config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, samples_per_step=1),
    )

    derived_path = derive_replica_bundle(
        parent_path,
        tmp_path,
        derived_config,
        repo_root=REPO_ROOT,
    )
    repeated_path = derive_replica_bundle(
        parent_path,
        tmp_path,
        derived_config,
        repo_root=REPO_ROOT,
    )
    parent = load_replica_bundle(parent_path)
    derived = load_replica_bundle_for_config(tmp_path, derived_config)

    assert repeated_path == derived_path
    assert derived.stream_plan == parent.stream_plan.prefix_per_step(1)
    assert derived.metadata["model_state_hash"] == parent.metadata[
        "model_state_hash"
    ]
    assert derived.metadata["partition_hash"] == parent.metadata["partition_hash"]
    provenance = derived.metadata["stream_derivation"]
    assert provenance["parent_bundle_id"] == parent.metadata["bundle_id"]
    assert provenance["parent_stream_plan_hash"] == parent.stream_plan.content_hash
    assert provenance["derived_stream_plan_hash"] == derived.stream_plan.content_hash
    assert provenance["requested_samples_per_step"] == 1
    assert provenance["prefix_rule"] == (
        "first_m_ordered_observations_at_each_p_step"
    )
    for name, tensor in derived.model.state_dict().items():
        assert torch.equal(tensor, parent.model.state_dict()[name])


def test_derived_bundle_rejects_non_stream_design_changes(tmp_path: Path) -> None:
    config, _, _, partitions, stream, model, layout = _tiny_fixture()
    initialization = InitializationResult(
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        wall_time_seconds=1.0,
        epochs_completed=1,
        stopped_on_target=False,
        history=(),
        final_metrics={},
    )
    parent_path = save_replica_bundle(
        tmp_path,
        config,
        model,
        layout,
        partitions,
        stream,
        initialization,
        device=torch.device("cpu"),
        repo_root=REPO_ROOT,
    )
    incompatible = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            samples_per_step=1,
            non_nine_sampling="balanced",
        ),
    )

    with pytest.raises(ReplicaBundleError, match="outside samples_per_step"):
        derive_replica_bundle(
            parent_path,
            tmp_path,
            incompatible,
            repo_root=REPO_ROOT,
        )


def test_scheduled_bundle_reuses_parent_initialization_and_common_uniforms(
    tmp_path: Path,
) -> None:
    config, _, _, partitions, stream, model, layout = _tiny_fixture()
    initialization = InitializationResult(
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
        wall_time_seconds=1.0,
        epochs_completed=1,
        stopped_on_target=False,
        history=(),
        final_metrics={"accuracy": 0.5},
    )
    parent_path = save_replica_bundle(
        tmp_path,
        config,
        model,
        layout,
        partitions,
        stream,
        initialization,
        device=torch.device("cpu"),
        repo_root=REPO_ROOT,
    )
    schedules = (
        ScheduleConfig(
            kind="linear",
            p_start=0.0,
            p_end=0.2,
            center_fraction=None,
            steepness=None,
        ),
        ScheduleConfig(
            kind="normalized_logistic",
            p_start=0.0,
            p_end=0.2,
            center_fraction=0.5,
            steepness=32.0,
        ),
    )
    derived = []
    for schedule in schedules:
        scheduled_config = dataclasses.replace(
            config,
            data=dataclasses.replace(config.data, schedule=schedule),
        )
        path = derive_scheduled_replica_bundle(
            parent_path,
            tmp_path,
            scheduled_config,
            torch.arange(100) % 10,
            repo_root=REPO_ROOT,
        )
        derived.append(load_replica_bundle_for_config(tmp_path, scheduled_config))
        assert derive_scheduled_replica_bundle(
            parent_path,
            tmp_path,
            scheduled_config,
            torch.arange(100) % 10,
            repo_root=REPO_ROOT,
        ) == path

    parent = load_replica_bundle(parent_path)
    assert len({item.stream_plan.schedule_hash for item in derived}) == 2
    assert len({item.stream_plan.uniform_stream_hash for item in derived}) == 1
    for item in derived:
        assert item.metadata["model_state_hash"] == parent.metadata["model_state_hash"]
        assert item.metadata["partition_hash"] == parent.metadata["partition_hash"]
        provenance = item.metadata["stream_derivation"]
        assert provenance["parent_bundle_id"] == parent.metadata["bundle_id"]
        assert provenance["derived_schedule_hash"] == item.stream_plan.schedule_hash
        assert provenance["uniform_stream_hash"] == item.stream_plan.uniform_stream_hash
        for name, tensor in item.model.state_dict().items():
            assert torch.equal(tensor, parent.model.state_dict()[name])


def test_tiny_initialization_is_deterministic() -> None:
    fixtures = [_tiny_fixture(), _tiny_fixture()]
    results = []
    states = []
    for (
        config,
        train_dataset,
        test_dataset,
        partitions,
        _,
        model,
        _,
    ) in fixtures:
        seeds = derive_seed_map(config.replica_seed)
        results.append(
            fit_p0_initialization(
                model,
                train_dataset,
                test_dataset,
                partitions,
                config.initialization,
                loader_seed=seeds["initialization_loader"],
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        )
        states.append(model.state_dict())

    assert results[0].epochs_completed == results[1].epochs_completed
    assert results[0].stopped_on_target == results[1].stopped_on_target
    assert results[0].history == results[1].history
    assert results[0].final_metrics == results[1].final_metrics
    for name in states[0]:
        assert torch.equal(states[0][name], states[1][name])
