"""Observed online-data exposure accounting for immutable stream plans."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


EXPOSURE_FIELDS = (
    "batch_consumed_by_optimizer",
    "batch_observation_count",
    "batch_nine_count",
    "batch_non_nine_count",
    "batch_unique_observation_count",
    "batch_unique_nine_observation_count",
    "batch_unique_non_nine_observation_count",
    "before_cumulative_observations",
    "before_cumulative_nine_observations",
    "before_cumulative_non_nine_observations",
    "before_cumulative_unique_observations",
    "before_cumulative_unique_nine_observations",
    "before_cumulative_unique_non_nine_observations",
    "after_cumulative_observations",
    "after_cumulative_nine_observations",
    "after_cumulative_non_nine_observations",
    "after_cumulative_unique_observations",
    "after_cumulative_unique_nine_observations",
    "after_cumulative_unique_non_nine_observations",
)


def stream_exposure_rows(
    observation_indices: Sequence[Sequence[int]],
    class_labels: Sequence[Sequence[int]],
    *,
    consume_final_batch: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Return exact before/after exposure counts for one ordered stream.

    Controller trajectories evaluate the final path point but do not optimize
    there, so the default leaves the final available batch unconsumed by the optimizer.
    """

    if len(observation_indices) != len(class_labels):
        raise ValueError("observation and label step counts differ")
    if not observation_indices:
        raise ValueError("stream plan must contain at least one step")

    cumulative = 0
    cumulative_nine = 0
    seen: set[int] = set()
    seen_nine: set[int] = set()
    seen_non_nine: set[int] = set()
    rows = []
    final_step = len(observation_indices) - 1

    for step, (indices_raw, labels_raw) in enumerate(
        zip(observation_indices, class_labels, strict=True)
    ):
        indices = tuple(int(value) for value in indices_raw)
        labels = tuple(int(value) for value in labels_raw)
        if len(indices) != len(labels):
            raise ValueError(f"observation and label counts differ at step {step}")
        if not indices:
            raise ValueError(f"stream batch is empty at step {step}")
        if any(label < 0 or label > 9 for label in labels):
            raise ValueError(f"invalid MNIST label at step {step}")

        batch_nine = sum(label == 9 for label in labels)
        batch_non_nine = len(labels) - batch_nine
        batch_unique = set(indices)
        batch_unique_nine = {
            index for index, label in zip(indices, labels, strict=True) if label == 9
        }
        batch_unique_non_nine = batch_unique - batch_unique_nine
        before = {
            "before_cumulative_observations": cumulative,
            "before_cumulative_nine_observations": cumulative_nine,
            "before_cumulative_non_nine_observations": cumulative - cumulative_nine,
            "before_cumulative_unique_observations": len(seen),
            "before_cumulative_unique_nine_observations": len(seen_nine),
            "before_cumulative_unique_non_nine_observations": len(seen_non_nine),
        }

        consumed = consume_final_batch or step < final_step
        if consumed:
            cumulative += len(indices)
            cumulative_nine += batch_nine
            seen.update(indices)
            seen_nine.update(batch_unique_nine)
            seen_non_nine.update(batch_unique_non_nine)

        rows.append(
            {
                "step": step,
                "batch_consumed_by_optimizer": consumed,
                "batch_observation_count": len(indices),
                "batch_nine_count": batch_nine,
                "batch_non_nine_count": batch_non_nine,
                "batch_unique_observation_count": len(batch_unique),
                "batch_unique_nine_observation_count": len(batch_unique_nine),
                "batch_unique_non_nine_observation_count": len(
                    batch_unique_non_nine
                ),
                **before,
                "after_cumulative_observations": cumulative,
                "after_cumulative_nine_observations": cumulative_nine,
                "after_cumulative_non_nine_observations": (
                    cumulative - cumulative_nine
                ),
                "after_cumulative_unique_observations": len(seen),
                "after_cumulative_unique_nine_observations": len(seen_nine),
                "after_cumulative_unique_non_nine_observations": len(
                    seen_non_nine
                ),
            }
        )
    return tuple(rows)
