from __future__ import annotations

import json
from pathlib import Path

import pytest

from mnist_experiment.run_replay import _active_indices
from src.config import ConfigError, ExperimentConfig
from src.replay import (
    CANONICAL_FIFO_METADATA_BYTES,
    CANONICAL_OBSERVATION_BYTES,
    FifoReplayBuffer,
    ReplayEvent,
    stream_events,
)


REPO_ROOT = Path(__file__).parents[2]
SOURCE_CONFIG = (
    REPO_ROOT
    / "mnist_experiment"
    / "configs"
    / "phase8_gpu_k50_fixed_rank8.json"
)


def _events(step: int, indices: tuple[int, ...]) -> tuple[ReplayEvent, ...]:
    return stream_events(
        step,
        indices,
        tuple(index % 10 for index in indices),
        samples_per_step=len(indices),
    )


def test_fifo_sequence_evicts_oldest_events_without_deduplicating_sources() -> None:
    replay = FifoReplayBuffer(capacity=3)
    first = _events(0, (7, 7))
    second = _events(1, (8, 9))

    assert replay.insert(first) == ()
    evicted = replay.insert(second)

    assert [event.event_id for event in evicted] == [0]
    assert [event.event_id for event in replay.events] == [1, 2, 3]
    assert [event.observation_index for event in replay.events] == [7, 8, 9]
    assert replay.total_insertions == 4
    assert replay.total_evictions == 1


def test_unbounded_replay_retains_every_arrival_exactly_once() -> None:
    replay = FifoReplayBuffer(capacity=None)
    for step in range(4):
        assert replay.insert(_events(step, (step, step))) == ()

    assert len(replay.events) == 8
    assert len({event.event_id for event in replay.events}) == 8
    assert replay.total_evictions == 0
    assert replay.logical_persistent_bytes == (
        CANONICAL_FIFO_METADATA_BYTES + 8 * CANONICAL_OBSERVATION_BYTES
    )


def test_replay_state_round_trip_preserves_fifo_and_ledger() -> None:
    replay = FifoReplayBuffer(capacity=2)
    replay.insert(_events(0, (1, 2)))
    replay.insert(_events(1, (3, 4)))

    restored = FifoReplayBuffer.from_mapping(replay.to_mapping())

    assert restored.to_mapping() == replay.to_mapping()
    assert restored.serialized_state_bytes > restored.physical_index_state_bytes


def test_active_block_keeps_repeated_source_draws_but_not_repeated_events() -> None:
    replay = _events(0, (7, 8))
    current = _events(1, (7, 9))

    assert _active_indices(current, replay) == (7, 9, 7, 8)
    with pytest.raises(RuntimeError, match="pre-update"):
        _active_indices(replay, replay)


def test_schema_thirteen_requires_valid_replay_configuration() -> None:
    raw = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    raw.update(
        {
            "schema_version": 13,
            "artifact_schema_version": 5,
            "metric_schema_version": 9,
            "replay": {"capacity": "unbounded", "policy": "fifo", "max_steps": 3},
        }
    )

    config = ExperimentConfig.from_mapping(raw)

    assert config.replay is not None
    assert config.replay.capacity == "unbounded"
    assert config.to_mapping() == raw

    raw["replay"]["capacity"] = -1
    with pytest.raises(ConfigError, match="capacity"):
        ExperimentConfig.from_mapping(raw)


def test_schema_fourteen_requires_hybrid_archive_source() -> None:
    raw = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    raw.update(
        {
            "schema_version": 14,
            "artifact_schema_version": 6,
            "metric_schema_version": 10,
            "replay": {
                "capacity": 8,
                "policy": "fifo",
                "max_steps": 3,
                "mode": "hybrid",
                "archive_initialization_artifact": "cache/source.pt",
            },
        }
    )

    config = ExperimentConfig.from_mapping(raw)

    assert config.replay is not None
    assert config.replay.mode == "hybrid"
    assert config.to_mapping() == raw

    raw["replay"]["archive_initialization_artifact"] = None
    with pytest.raises(ConfigError, match="archive_initialization_artifact"):
        ExperimentConfig.from_mapping(raw)
