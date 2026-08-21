"""Deterministic event-level FIFO replay for Plan 3."""

from __future__ import annotations

import dataclasses
import io
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch


REPLAY_STATE_SCHEMA_VERSION = 1
CANONICAL_OBSERVATION_BYTES = 800
CANONICAL_FIFO_METADATA_BYTES = 24
PHYSICAL_EVENT_STATE_BYTES = 5 * 8


@dataclasses.dataclass(frozen=True)
class ReplayEvent:
    event_id: int
    stream_step: int
    within_step: int
    observation_index: int
    class_label: int

    def validate(self) -> None:
        for name in (
            "event_id",
            "stream_step",
            "within_step",
            "observation_index",
            "class_label",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"replay event {name} must be nonnegative")
        if self.class_label > 9:
            raise ValueError("replay event class_label must be in [0, 9]")

    def to_mapping(self) -> dict[str, int]:
        return dataclasses.asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReplayEvent":
        event = cls(
            event_id=value["event_id"],
            stream_step=value["stream_step"],
            within_step=value["within_step"],
            observation_index=value["observation_index"],
            class_label=value["class_label"],
        )
        event.validate()
        return event


def stream_events(
    step: int,
    observation_indices: Sequence[int],
    class_labels: Sequence[int],
    *,
    samples_per_step: int,
) -> tuple[ReplayEvent, ...]:
    if len(observation_indices) != len(class_labels):
        raise ValueError("observation indices and labels must have equal length")
    if len(observation_indices) != samples_per_step:
        raise ValueError("stream event count does not match samples_per_step")
    events = tuple(
        ReplayEvent(
            event_id=step * samples_per_step + within_step,
            stream_step=step,
            within_step=within_step,
            observation_index=int(observation_index),
            class_label=int(class_label),
        )
        for within_step, (observation_index, class_label) in enumerate(
            zip(observation_indices, class_labels, strict=True)
        )
    )
    for event in events:
        event.validate()
    return events


@dataclasses.dataclass
class FifoReplayBuffer:
    capacity: int | None
    _events: list[ReplayEvent] = dataclasses.field(default_factory=list)
    total_insertions: int = 0
    total_evictions: int = 0

    def __post_init__(self) -> None:
        if self.capacity is not None and (
            not isinstance(self.capacity, int)
            or isinstance(self.capacity, bool)
            or self.capacity < 0
        ):
            raise ValueError("replay capacity must be nonnegative or unbounded")
        self._validate_state()

    @property
    def events(self) -> tuple[ReplayEvent, ...]:
        return tuple(self._events)

    @property
    def capacity_mapping(self) -> int | str:
        return "unbounded" if self.capacity is None else self.capacity

    @property
    def logical_persistent_bytes(self) -> int:
        return (
            CANONICAL_FIFO_METADATA_BYTES
            + len(self._events) * CANONICAL_OBSERVATION_BYTES
        )

    @property
    def physical_index_state_bytes(self) -> int:
        return (
            CANONICAL_FIFO_METADATA_BYTES
            + len(self._events) * PHYSICAL_EVENT_STATE_BYTES
        )

    @property
    def serialized_state_bytes(self) -> int:
        buffer = io.BytesIO()
        torch.save(self.to_mapping(), buffer)
        return buffer.tell()

    def insert(self, events: Iterable[ReplayEvent]) -> tuple[ReplayEvent, ...]:
        inserted = tuple(events)
        for event in inserted:
            event.validate()
        existing_ids = {event.event_id for event in self._events}
        inserted_ids = [event.event_id for event in inserted]
        if len(inserted_ids) != len(set(inserted_ids)):
            raise ValueError("one replay insertion cannot repeat an event identity")
        if existing_ids.intersection(inserted_ids):
            raise ValueError("replay event identity was inserted more than once")

        self._events.extend(inserted)
        self.total_insertions += len(inserted)
        overflow = (
            0
            if self.capacity is None
            else max(0, len(self._events) - self.capacity)
        )
        evicted = tuple(self._events[:overflow])
        if overflow:
            del self._events[:overflow]
            self.total_evictions += overflow
        self._validate_state()
        return evicted

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": REPLAY_STATE_SCHEMA_VERSION,
            "policy": "fifo",
            "capacity": self.capacity_mapping,
            "total_insertions": self.total_insertions,
            "total_evictions": self.total_evictions,
            "events": [event.to_mapping() for event in self._events],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FifoReplayBuffer":
        if value.get("schema_version") != REPLAY_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported replay state schema")
        if value.get("policy") != "fifo":
            raise ValueError("replay state policy must be fifo")
        capacity_value = value.get("capacity")
        capacity = None if capacity_value == "unbounded" else capacity_value
        events_value = value.get("events")
        if not isinstance(events_value, list):
            raise ValueError("replay state events must be a list")
        replay = cls(
            capacity=capacity,
            _events=[ReplayEvent.from_mapping(event) for event in events_value],
            total_insertions=value.get("total_insertions"),
            total_evictions=value.get("total_evictions"),
        )
        replay._validate_state()
        return replay

    def _validate_state(self) -> None:
        if not isinstance(self.total_insertions, int) or self.total_insertions < 0:
            raise ValueError("total_insertions must be nonnegative")
        if not isinstance(self.total_evictions, int) or self.total_evictions < 0:
            raise ValueError("total_evictions must be nonnegative")
        if self.capacity is not None and len(self._events) > self.capacity:
            raise ValueError("replay state exceeds its capacity")
        event_ids = []
        for event in self._events:
            event.validate()
            event_ids.append(event.event_id)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("replay state repeats an event identity")
        if event_ids != sorted(event_ids):
            raise ValueError("replay state is not in FIFO event order")
        if self.total_insertions - self.total_evictions != len(self._events):
            raise ValueError("replay insertion and eviction counts are inconsistent")
