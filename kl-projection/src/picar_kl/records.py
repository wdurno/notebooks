"""JSON-friendly records shared by phase 1 and phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .actions import (
    ActionDistribution,
    ActionVector,
    action_distribution_to_vector,
    action_name_to_distribution,
    distribution_to_action_name,
    validate_action_distribution,
    validate_action_vector,
)
from .latency import LatencyEvent


@dataclass(frozen=True)
class ActionRecord:
    """One action decision expressed as probabilities and robot control."""

    distribution: ActionDistribution
    action_name: str
    executed_vector: ActionVector
    generated_text: str = ""
    source: str = "vlm"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_action_name(
        cls,
        action_name: str,
        *,
        generated_text: str = "",
        executed_vector: dict[str, float] | None = None,
        source: str = "vlm",
        metadata: dict[str, Any] | None = None,
    ) -> "ActionRecord":
        distribution = action_name_to_distribution(action_name)
        vector = executed_vector or action_distribution_to_vector(distribution)
        return cls(
            distribution=distribution,
            action_name=action_name,
            executed_vector=validate_action_vector(vector),
            generated_text=generated_text,
            source=source,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_distribution(
        cls,
        distribution: list[float] | tuple[float, ...],
        *,
        generated_text: str = "",
        executed_vector: dict[str, float] | None = None,
        source: str = "lstm",
        metadata: dict[str, Any] | None = None,
    ) -> "ActionRecord":
        checked_distribution = validate_action_distribution(distribution)
        vector = executed_vector or action_distribution_to_vector(checked_distribution)
        return cls(
            distribution=checked_distribution,
            action_name=distribution_to_action_name(checked_distribution),
            executed_vector=validate_action_vector(vector),
            generated_text=generated_text,
            source=source,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionRecord":
        return cls(
            distribution=validate_action_distribution(payload["distribution"]),
            action_name=str(payload["action_name"]),
            executed_vector=validate_action_vector(payload["executed_vector"]),
            generated_text=str(payload.get("generated_text") or ""),
            source=str(payload.get("source") or "unknown"),
            metadata=dict(payload.get("metadata") or {}),
        )

    @classmethod
    def from_legacy_dict(cls, payload: dict[str, Any]) -> "ActionRecord":
        action_name = str(payload["agentic_action_name"])
        executed_vector = payload.get("executed_action_vector")
        generated_text = str(payload.get("generated_text") or "")
        metadata = {
            "agentic_action_vector": payload.get("agentic_action_vector"),
            "actor_action_vector": payload.get("actor_action_vector"),
            "critic_value": payload.get("critic_value"),
            "logp_beta_sum": payload.get("logp_beta_sum"),
        }
        return cls.from_action_name(
            action_name,
            generated_text=generated_text,
            executed_vector=executed_vector,
            source="legacy-vlm",
            metadata={key: value for key, value in metadata.items() if value is not None},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "distribution": list(self.distribution),
            "action_name": self.action_name,
            "executed_vector": dict(self.executed_vector),
            "generated_text": self.generated_text,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class Phase1ObservationRecord:
    """One persisted phase 1 observation row."""

    run_uuid: str | None
    run_dir: Path | None
    timestamp: str | None
    source: str
    step_index: int
    image_path: Path | None
    messages: list[dict[str, Any]]
    user_texts: list[str]
    action: ActionRecord | None
    reward: float | None = None
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_events: tuple[LatencyEvent, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        run_uuid: str | None = None,
        run_dir: Path | None = None,
    ) -> "Phase1ObservationRecord":
        action_payload = payload.get("action")
        action = ActionRecord.from_dict(action_payload) if action_payload else None
        return cls(
            run_uuid=run_uuid,
            run_dir=Path(run_dir) if run_dir is not None else None,
            timestamp=payload.get("timestamp"),
            source=str(payload.get("source") or "unknown"),
            step_index=int(payload["step_index"]),
            image_path=Path(payload["image_path"]) if payload.get("image_path") else None,
            messages=list(payload.get("messages") or []),
            user_texts=[str(item) for item in payload.get("user_texts") or []],
            action=action,
            reward=None if payload.get("reward") is None else float(payload["reward"]),
            done=bool(payload.get("done", False)),
            metadata=dict(payload.get("metadata") or {}),
            latency_events=tuple(
                LatencyEvent.from_dict(event) for event in payload.get("latency_events") or ()
            ),
            raw=dict(payload),
        )

    @classmethod
    def from_legacy_dict(
        cls,
        payload: dict[str, Any],
        *,
        run_uuid: str | None = None,
        run_dir: Path | None = None,
    ) -> "Phase1ObservationRecord":
        action_payload = payload.get("action")
        action = ActionRecord.from_legacy_dict(action_payload) if action_payload else None
        return cls(
            run_uuid=run_uuid,
            run_dir=Path(run_dir) if run_dir is not None else None,
            timestamp=payload.get("timestamp"),
            source=str(payload.get("source") or "legacy"),
            step_index=int(payload["step_index"]),
            image_path=Path(payload["image_path"]) if payload.get("image_path") else None,
            messages=list(payload.get("messages") or []),
            user_texts=[str(item) for item in payload.get("user_texts") or []],
            action=action,
            reward=None if payload.get("reward") is None else float(payload["reward"]),
            done=bool(payload.get("done", False)),
            metadata=dict(payload.get("metadata") or {}),
            latency_events=(),
            raw=dict(payload),
        )

    @property
    def image_file(self) -> Path | None:
        if self.image_path is None:
            return None
        if self.image_path.is_absolute() or self.run_dir is None:
            return self.image_path
        return self.run_dir / self.image_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_uuid": self.run_uuid,
            "timestamp": self.timestamp,
            "source": self.source,
            "step_index": self.step_index,
            "image_path": None if self.image_path is None else str(self.image_path),
            "messages": list(self.messages),
            "user_texts": list(self.user_texts),
            "action": None if self.action is None else self.action.to_dict(),
            "reward": self.reward,
            "done": self.done,
            "metadata": dict(self.metadata),
            "latency_events": [event.to_dict() for event in self.latency_events],
        }
