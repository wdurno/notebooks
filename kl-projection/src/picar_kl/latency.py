"""Raw wall-clock latency records."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any


@dataclass(frozen=True)
class LatencyEvent:
    """One measured interval, stored without aggregation assumptions."""

    name: str
    started_at: float
    ended_at: float
    duration_seconds: float
    clock: str = "time.time"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_bounds(
        cls,
        name: str,
        *,
        started_at: float,
        ended_at: float,
        clock: str = "time.time",
        metadata: dict[str, Any] | None = None,
    ) -> "LatencyEvent":
        duration_seconds = float(ended_at) - float(started_at)
        if duration_seconds < 0.0:
            raise ValueError("Latency event ended before it started")
        return cls(
            name=str(name),
            started_at=float(started_at),
            ended_at=float(ended_at),
            duration_seconds=duration_seconds,
            clock=clock,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LatencyEvent":
        return cls(
            name=str(payload["name"]),
            started_at=float(payload["started_at"]),
            ended_at=float(payload["ended_at"]),
            duration_seconds=float(payload["duration_seconds"]),
            clock=str(payload.get("clock", "time.time")),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "clock": self.clock,
            "metadata": dict(self.metadata),
        }


class LatencyTimer:
    """Context manager for recording one latency event."""

    def __init__(self, name: str, *, metadata: dict[str, Any] | None = None):
        self.name = name
        self.metadata = dict(metadata or {})
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.event: LatencyEvent | None = None

    def __enter__(self) -> "LatencyTimer":
        self.started_at = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc, traceback
        self.ended_at = time.time()
        if self.started_at is None:
            raise RuntimeError("Latency timer exited before starting")
        self.event = LatencyEvent.from_bounds(
            self.name,
            started_at=self.started_at,
            ended_at=self.ended_at,
            metadata=self.metadata,
        )
        return False
