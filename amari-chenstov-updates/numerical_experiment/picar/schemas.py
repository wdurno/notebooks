"""Small JSON contracts shared by storage, learner, and artifact-only reports."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from . import ARTIFACT_SCHEMA_VERSION, METRIC_SCHEMA_VERSION


METRIC_COLUMNS = (
    "schema_version", "phase", "condition", "replica", "step", "episode",
    "new_transitions", "replay_exposures", "retired_transitions", "model_version",
    "data_nll", "ewc", "ridge", "fd", "total_objective",
    "update_seconds", "inference_seconds", "fisher_seconds", "active_seconds",
    "deadline_misses", "success", "collision", "end_reason", "drift_alpha",
    "test_domain", "test_success", "test_count", "smoke_only",
)


def metric_row(**values: Any) -> dict[str, Any]:
    unknown = set(values) - set(METRIC_COLUMNS)
    if unknown:
        raise ValueError(f"unknown metric fields: {sorted(unknown)}")
    row = {name: None for name in METRIC_COLUMNS}
    row.update(values)
    row["schema_version"] = METRIC_SCHEMA_VERSION
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"nonfinite metric {key}")
    return row


@dataclass(frozen=True)
class TransitionRecord:
    transition_id: str
    episode_id: str
    arrival_index: int
    observation_ref: str
    next_observation_ref: str
    context_refs: tuple[str, ...]
    instruction: str
    action_logits: tuple[float, ...]
    action_probabilities: tuple[float, ...]
    requested_control: dict[str, float]
    executed_control: dict[str, float]
    baseline: float
    model_version: int
    elapsed_s: float
    reward: float
    discount: float
    terminated: bool
    truncated: bool
    end_reason: str | None
    receipt: dict[str, Any] = field(default_factory=dict)
    instruction_events: tuple[dict[str, Any], ...] = ()
    action_history: tuple[tuple[float, ...], ...] = ()
    context_arrivals: tuple[int, ...] = ()
    context_elapsed: tuple[float, ...] = ()
    context_start_s: float = 0.0
    return_value: float | None = None
    return_bin: int | None = None
    behavior_kind: str = "gaussian_exploration"
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported transition schema")
        if not all(isinstance(s, str) and s for s in (
            self.transition_id, self.episode_id, self.observation_ref, self.next_observation_ref
        )):
            raise ValueError("transition/frame identifiers must be nonempty")
        if type(self.arrival_index) is not int or self.arrival_index < 0:
            raise ValueError("arrival_index must be a nonnegative integer")
        if len(self.action_logits) != 7 or len(self.action_probabilities) != 8:
            raise ValueError("seven logits and eight probabilities required")
        scalars = (*self.action_logits, *self.action_probabilities, self.baseline,
                   self.elapsed_s, self.reward, self.discount)
        if not all(math.isfinite(x) for x in scalars):
            raise ValueError("transition contains a nonfinite numerical value")
        if min(self.action_probabilities) < 0 or not math.isclose(sum(self.action_probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("action probabilities must form a distribution")
        if self.elapsed_s <= 0 or not 0 < self.discount <= 1:
            raise ValueError("invalid transition clock or discount")
        if self.terminated and self.truncated:
            raise ValueError("a task termination is not an infrastructure truncation")
        if self.context_arrivals or self.context_elapsed:
            if not (len(self.context_refs) == len(self.action_history) ==
                    len(self.context_arrivals) == len(self.context_elapsed)):
                raise ValueError("raw history fields must have equal lengths")
            if any(n > self.arrival_index or n < 0 for n in self.context_arrivals):
                raise ValueError("history cannot include future observations")
            if any(not math.isfinite(t) or t < 0 for t in self.context_elapsed):
                raise ValueError("history elapsed times must be finite and nonnegative")
            if not math.isfinite(self.context_start_s) or self.context_start_s < 0:
                raise ValueError("invalid history start time")
        if (self.return_value is None) != (self.return_bin is None):
            raise ValueError("return value and bin must become ready together")
        if self.return_value is not None:
            if self.truncated or not math.isfinite(self.return_value):
                raise ValueError("truncated/nonfinite returns cannot train")
            if type(self.return_bin) is not int or self.return_bin < 0:
                raise ValueError("return bin must be a nonnegative integer")

    @property
    def ready(self) -> bool:
        return self.return_value is not None

    def labeled(self, value: float, bin_index: int) -> TransitionRecord:
        if self.ready:
            raise ValueError("a finalized return label cannot be overwritten")
        return replace(self, return_value=float(value), return_bin=int(bin_index))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> TransitionRecord:
        data = dict(value)
        for name in ("context_refs", "action_logits", "action_probabilities", "instruction_events",
                     "context_arrivals", "context_elapsed"):
            if name in data:
                data[name] = tuple(data[name])
        if "action_history" in data:
            data["action_history"] = tuple(tuple(x) for x in data["action_history"])
        return cls(**data)
