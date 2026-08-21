"""Deterministic environmental schedules and trajectory metadata."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

from .config import DataConfig, ScheduleConfig

SCHEDULE_SCHEMA_VERSION = 1
SCHEDULE_TRAJECTORY_SCHEMA_VERSION = 1


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class ResolvedSchedule:
    name: str
    kind: str
    p_start: float
    p_end: float
    center_fraction: float | None
    center_p: float | None
    steepness: float | None
    p_values: tuple[float, ...]
    delta_p_values: tuple[float, ...]
    max_speed_transition: int
    schema_version: int = SCHEDULE_SCHEMA_VERSION

    def to_mapping(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def content_hash(self) -> str:
        return _canonical_hash(self.to_mapping())


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def resolve_schedule(config: DataConfig) -> ResolvedSchedule:
    """Resolve the legacy path or one explicit schedule into stored values."""

    config.validate()
    schedule = config.schedule
    if schedule is None:
        schedule = ScheduleConfig(
            kind="linear",
            p_start=0.0,
            p_end=1.0,
            center_fraction=None,
            steepness=None,
        )
        name = "legacy_linear_0_1"
    else:
        name = schedule.kind

    coordinates = tuple(
        step / (config.num_p_steps - 1)
        for step in range(config.num_p_steps)
    )
    if schedule.kind == "linear":
        scaled = coordinates
        center_p = None
    else:
        assert schedule.center_fraction is not None
        assert schedule.steepness is not None
        lower = _sigmoid(-schedule.steepness * schedule.center_fraction)
        upper = _sigmoid(
            schedule.steepness * (1.0 - schedule.center_fraction)
        )
        denominator = upper - lower
        if denominator <= 0.0 or not math.isfinite(denominator):
            raise ValueError("normalized logistic schedule is numerically degenerate")
        scaled = tuple(
            (
                _sigmoid(
                    schedule.steepness
                    * (coordinate - schedule.center_fraction)
                )
                - lower
            )
            / denominator
            for coordinate in coordinates
        )
        center_scaled = (0.5 - lower) / denominator
        center_p = schedule.p_start + (
            schedule.p_end - schedule.p_start
        ) * center_scaled

    p_values = tuple(
        schedule.p_start
        + (schedule.p_end - schedule.p_start) * value
        for value in scaled
    )
    # Pin endpoints so artifact identities do not depend on roundoff in sigmoid.
    p_values = (schedule.p_start, *p_values[1:-1], schedule.p_end)
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in p_values
    ):
        raise ValueError("resolved schedule contains an invalid prevalence")
    if any(right < left for left, right in zip(p_values, p_values[1:])):
        raise ValueError("resolved schedule must be nondecreasing")
    if not any(right > left for left, right in zip(p_values, p_values[1:])):
        raise ValueError("resolved schedule must contain positive movement")
    delta_p_values = (0.0,) + tuple(
        right - left for left, right in zip(p_values, p_values[1:])
    )
    max_speed_transition = max(
        range(1, len(delta_p_values)),
        key=delta_p_values.__getitem__,
    )
    return ResolvedSchedule(
        name=name,
        kind=schedule.kind,
        p_start=schedule.p_start,
        p_end=schedule.p_end,
        center_fraction=schedule.center_fraction,
        center_p=center_p,
        steepness=schedule.steepness,
        p_values=p_values,
        delta_p_values=delta_p_values,
        max_speed_transition=max_speed_transition,
    )


def schedule_trajectory_mapping(
    schedule: ResolvedSchedule,
    class_labels: Sequence[Sequence[int]],
    *,
    samples_per_step: int,
    stream_plan_hash: str,
    uniform_stream_hash: str | None,
    consume_final_batch: bool = False,
) -> dict[str, Any]:
    """Build an independently versioned exposure record for one path."""

    if len(class_labels) != len(schedule.p_values):
        raise ValueError("schedule and class-label step counts differ")
    if samples_per_step < 1:
        raise ValueError("samples_per_step must be positive")

    expected_available = 0.0
    expected_consumed = 0.0
    realized_available = 0
    realized_consumed = 0
    rows = []
    final_step = len(schedule.p_values) - 1
    for step, (p_value, delta_p, labels) in enumerate(
        zip(
            schedule.p_values,
            schedule.delta_p_values,
            class_labels,
            strict=True,
        )
    ):
        if len(labels) != samples_per_step:
            raise ValueError(f"class-label count differs at step {step}")
        expected_batch = samples_per_step * p_value
        realized_batch = sum(int(label) == 9 for label in labels)
        expected_available += expected_batch
        realized_available += realized_batch
        consumed = consume_final_batch or step < final_step
        if consumed:
            expected_consumed += expected_batch
            realized_consumed += realized_batch
        rows.append(
            {
                "step": step,
                "p": p_value,
                "delta_p": delta_p,
                "expected_batch_nines": expected_batch,
                "realized_batch_nines": realized_batch,
                "batch_consumed_by_optimizer": consumed,
                "expected_cumulative_nines_available": expected_available,
                "realized_cumulative_nines_available": realized_available,
                "expected_cumulative_nines_consumed": expected_consumed,
                "realized_cumulative_nines_consumed": realized_consumed,
            }
        )
    return {
        "schema_version": SCHEDULE_TRAJECTORY_SCHEMA_VERSION,
        "schedule": schedule.to_mapping(),
        "schedule_hash": schedule.content_hash,
        "stream_plan_hash": stream_plan_hash,
        "uniform_stream_hash": uniform_stream_hash,
        "samples_per_step": samples_per_step,
        "expected_total_observations_available": (
            samples_per_step * len(schedule.p_values)
        ),
        "expected_total_nines_available": expected_available,
        "realized_total_nines_available": realized_available,
        "expected_total_nines_consumed": expected_consumed,
        "realized_total_nines_consumed": realized_consumed,
        "max_speed_transition": schedule.max_speed_transition,
        "rows": rows,
    }
