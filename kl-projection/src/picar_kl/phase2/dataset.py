"""Phase 2 token-stream dataset construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from picar_kl.actions import action_count, action_name_to_distribution, validate_action_distribution
from picar_kl.data.phase1 import iter_phase1_records
from picar_kl.phase2.cache import VisualEncodingCache
from picar_kl.records import Phase1ObservationRecord


class Phase2DatasetError(RuntimeError):
    pass


@dataclass(frozen=True)
class Phase2StepExample:
    """One robot step resolved into cached visual tokens and action features."""

    visual_tokens: np.ndarray
    previous_action: tuple[float, ...]
    target_distribution: tuple[float, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_shape(self) -> tuple[int, int]:
        return tuple(int(item) for item in self.visual_tokens.shape)


@dataclass(frozen=True)
class Phase2SequenceExample:
    """A contiguous record sequence for token-stream LSTM training."""

    steps: tuple[Phase2StepExample, ...]

    @property
    def visual_token_shape(self) -> tuple[int, int]:
        if not self.steps:
            raise Phase2DatasetError("empty phase 2 sequence has no visual token shape")
        return self.steps[0].token_shape

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class Phase2Batch:
    """Padded token-stream batch before model-specific projection."""

    visual_tokens: np.ndarray
    previous_actions: np.ndarray
    target_distributions: np.ndarray
    readout_mask: np.ndarray
    step_mask: np.ndarray
    metadata: list[list[dict[str, Any]]]


@dataclass(frozen=True)
class Phase2WindowExample:
    """A prefix segment used to condition a later target segment."""

    prefix: Phase2SequenceExample
    target: Phase2SequenceExample


@dataclass(frozen=True)
class Phase2WindowBatch:
    """Batched prefix and target segments for joint Dh/LSTM training."""

    prefix: Phase2Batch
    target: Phase2Batch


def load_phase2_sequences(
    roots: Sequence[Path],
    *,
    cache: VisualEncodingCache,
    sequence_length: int,
    initial_previous_action: Sequence[float] | None = None,
    allow_missing_cached_encodings: bool = False,
) -> list[Phase2SequenceExample]:
    if int(sequence_length) < 1:
        raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
    sequences: list[Phase2SequenceExample] = []
    initial_action = _initial_previous_action(initial_previous_action)
    for root in roots:
        current_run: str | None = None
        run_steps: list[Phase2StepExample] = []
        previous_action = initial_action
        for record in iter_phase1_records(Path(root)):
            if record.run_uuid != current_run:
                sequences.extend(_chunk_steps(run_steps, sequence_length=int(sequence_length)))
                run_steps = []
                current_run = record.run_uuid
                previous_action = initial_action
            if record.action is None:
                continue
            try:
                step = phase2_step_from_record(
                    record,
                    cache=cache,
                    previous_action=previous_action,
                )
            except Phase2DatasetError as exc:
                if allow_missing_cached_encodings and "missing cached visual encoding" in str(exc):
                    continue
                raise
            run_steps.append(step)
            previous_action = step.target_distribution
        sequences.extend(_chunk_steps(run_steps, sequence_length=int(sequence_length)))
    return sequences


def load_phase2_windows(
    roots: Sequence[Path],
    *,
    cache: VisualEncodingCache,
    context_steps: int,
    prediction_steps: int,
    stride: int | None = None,
    initial_previous_action: Sequence[float] | None = None,
    allow_missing_cached_encodings: bool = False,
) -> list[Phase2WindowExample]:
    if int(context_steps) < 1:
        raise ValueError("context_steps must be >= 1")
    if int(prediction_steps) < 1:
        raise ValueError("prediction_steps must be >= 1")
    if stride is None:
        stride = int(prediction_steps)
    if int(stride) < 1:
        raise ValueError("stride must be >= 1")
    windows: list[Phase2WindowExample] = []
    initial_action = _initial_previous_action(initial_previous_action)
    for root in roots:
        current_run: str | None = None
        run_steps: list[Phase2StepExample] = []
        previous_action = initial_action
        for record in iter_phase1_records(Path(root)):
            if record.run_uuid != current_run:
                windows.extend(
                    _window_steps(
                        run_steps,
                        context_steps=int(context_steps),
                        prediction_steps=int(prediction_steps),
                        stride=int(stride),
                    )
                )
                run_steps = []
                current_run = record.run_uuid
                previous_action = initial_action
            if record.action is None:
                continue
            try:
                step = phase2_step_from_record(
                    record,
                    cache=cache,
                    previous_action=previous_action,
                )
            except Phase2DatasetError as exc:
                if allow_missing_cached_encodings and "missing cached visual encoding" in str(exc):
                    previous_action = record.action.distribution
                    continue
                raise
            run_steps.append(step)
            previous_action = step.target_distribution
        windows.extend(
            _window_steps(
                run_steps,
                context_steps=int(context_steps),
                prediction_steps=int(prediction_steps),
                stride=int(stride),
            )
        )
    return windows


def phase2_step_from_record(
    record: Phase1ObservationRecord,
    *,
    cache: VisualEncodingCache,
    previous_action: Sequence[float],
) -> Phase2StepExample:
    if record.action is None:
        raise Phase2DatasetError(f"record has no action: run={record.run_uuid} step={record.step_index}")
    try:
        encoding = cache.load(record)
    except FileNotFoundError as exc:
        raise Phase2DatasetError(
            f"missing cached visual encoding for run={record.run_uuid} step={record.step_index}"
        ) from exc
    target = validate_action_distribution(record.action.distribution)
    previous = validate_action_distribution(previous_action)
    return Phase2StepExample(
        visual_tokens=np.asarray(encoding.tokens),
        previous_action=previous,
        target_distribution=target,
        metadata={
            "run_uuid": record.run_uuid,
            "step_index": int(record.step_index),
            "source": record.source,
            "action_name": record.action.action_name,
            "image_path": None if record.image_path is None else str(record.image_path),
            "encoding_shape": list(encoding.shape),
            "missing_context_fields": _missing_context_fields(record),
        },
    )


def collate_phase2_sequences(sequences: Sequence[Phase2SequenceExample]) -> Phase2Batch:
    if not sequences:
        raise Phase2DatasetError("cannot collate an empty phase 2 batch")
    first_shape = sequences[0].visual_token_shape
    for sequence in sequences:
        for step in sequence.steps:
            if step.token_shape != first_shape:
                raise Phase2DatasetError(
                    f"incompatible visual token shape {step.token_shape}; expected {first_shape}"
                )
    batch_size = len(sequences)
    max_steps = max(sequence.step_count for sequence in sequences)
    num_tokens, visual_dim = first_shape
    action_dim = action_count()
    visual_tokens = np.zeros((batch_size, max_steps, num_tokens, visual_dim), dtype=sequences[0].steps[0].visual_tokens.dtype)
    previous_actions = np.zeros((batch_size, max_steps, action_dim), dtype=np.float32)
    targets = np.zeros((batch_size, max_steps, action_dim), dtype=np.float32)
    readout_mask = np.zeros((batch_size, max_steps), dtype=bool)
    step_mask = np.zeros((batch_size, max_steps), dtype=bool)
    metadata: list[list[dict[str, Any]]] = []

    for batch_idx, sequence in enumerate(sequences):
        row_meta: list[dict[str, Any]] = []
        for step_idx, step in enumerate(sequence.steps):
            visual_tokens[batch_idx, step_idx] = step.visual_tokens
            previous_actions[batch_idx, step_idx] = np.asarray(step.previous_action, dtype=np.float32)
            targets[batch_idx, step_idx] = np.asarray(step.target_distribution, dtype=np.float32)
            readout_mask[batch_idx, step_idx] = True
            step_mask[batch_idx, step_idx] = True
            row_meta.append(dict(step.metadata))
        metadata.append(row_meta)
    return Phase2Batch(
        visual_tokens=visual_tokens,
        previous_actions=previous_actions,
        target_distributions=targets,
        readout_mask=readout_mask,
        step_mask=step_mask,
        metadata=metadata,
    )


def collate_phase2_windows(windows: Sequence[Phase2WindowExample]) -> Phase2WindowBatch:
    if not windows:
        raise Phase2DatasetError("cannot collate an empty phase 2 window batch")
    return Phase2WindowBatch(
        prefix=collate_phase2_sequences([window.prefix for window in windows]),
        target=collate_phase2_sequences([window.target for window in windows]),
    )


def _chunk_steps(steps: list[Phase2StepExample], *, sequence_length: int) -> list[Phase2SequenceExample]:
    if not steps:
        return []
    return [
        Phase2SequenceExample(steps=tuple(steps[idx : idx + sequence_length]))
        for idx in range(0, len(steps), sequence_length)
    ]


def _window_steps(
    steps: list[Phase2StepExample],
    *,
    context_steps: int,
    prediction_steps: int,
    stride: int,
) -> list[Phase2WindowExample]:
    required = int(context_steps) + int(prediction_steps)
    if len(steps) < required:
        return []
    windows = []
    for idx in range(0, len(steps) - required + 1, int(stride)):
        prefix = Phase2SequenceExample(steps=tuple(steps[idx : idx + int(context_steps)]))
        target = Phase2SequenceExample(steps=tuple(steps[idx + int(context_steps) : idx + required]))
        windows.append(Phase2WindowExample(prefix=prefix, target=target))
    return windows


def _initial_previous_action(value: Sequence[float] | None) -> tuple[float, ...]:
    if value is not None:
        return validate_action_distribution(value)
    return action_name_to_distribution("look-forward")


def _missing_context_fields(record: Phase1ObservationRecord) -> list[str]:
    missing = []
    if "context" not in record.metadata:
        missing.append("context")
    if "reward_result" not in record.metadata:
        missing.append("reward_result")
    if not record.latency_events:
        missing.append("latency_events")
    return missing
