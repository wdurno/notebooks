"""VLM-facing phase 1 control messages and action parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from picar_kl.actions import (
    ActionDistribution,
    ACTION_NAMES,
    action_name_to_distribution,
    normalize_action_name,
    validate_action_distribution,
)


@dataclass(frozen=True)
class VLMDecision:
    """One VLM sparse-control decision."""

    action_distribution: ActionDistribution
    generated_text: str = ""
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_action_name(
        cls,
        action_name: str,
        *,
        generated_text: str = "",
        raw_response: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "VLMDecision":
        return cls(
            action_distribution=action_name_to_distribution(action_name),
            generated_text=generated_text,
            raw_response=raw_response,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_distribution(
        cls,
        distribution: list[float] | tuple[float, ...],
        *,
        generated_text: str = "",
        raw_response: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "VLMDecision":
        return cls(
            action_distribution=validate_action_distribution(distribution),
            generated_text=generated_text,
            raw_response=raw_response,
            metadata=dict(metadata or {}),
        )


def build_phase1_messages(
    *,
    task_prompt: str,
    step_index: int,
    user_texts: list[str],
    last_generated_text: str = "",
) -> list[dict[str, Any]]:
    user_text = "\n".join(text for text in user_texts if text.strip()) or "<none>"
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Control the PiCar robot.\n"
                        f"Primary task: {task_prompt}.\n"
                        "Choose one action and optionally say a short status update.\n"
                        "Respond as JSON: {\"action\": <action>, \"say\": <text>}.\n"
                        f"Allowed actions: {', '.join(ACTION_NAMES)}."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"step_index={int(step_index)}\n"
                        f"operator_speech={user_text}\n"
                        f"previous_robot_speech={last_generated_text or '<none>'}"
                    ),
                }
            ],
        },
    ]


def parse_action_response(text: str, *, default_action: str = "look-forward") -> VLMDecision:
    raw_response = str(text or "")
    payload = _extract_json_object(raw_response)
    if payload is not None:
        action_name = normalize_action_name(str(payload.get("action") or ""), default_action=default_action)
        generated_text = str(payload.get("say") or "")
        return VLMDecision.from_action_name(
            action_name,
            generated_text=generated_text,
            raw_response=raw_response,
            metadata={"parser": "json"},
        )
    action_name = normalize_action_name(raw_response, default_action=default_action)
    return VLMDecision.from_action_name(
        action_name,
        generated_text="",
        raw_response=raw_response,
        metadata={"parser": "text"},
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    first = text.find("{")
    last = text.rfind("}")
    if first < 0 or last < first:
        return None
    try:
        payload = json.loads(text[first : last + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
