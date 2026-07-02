"""Shared VLM message builders."""

from __future__ import annotations

import json
from typing import Any

from picar_kl.actions import ACTION_NAMES


CONTROL_SYSTEM_PROMPT = (
    "Control the PiCar robot.\n"
    "Choose one action and optionally say a short status update.\n"
    "The latest operator speech overrides the visual goal and previous actions. "
    "If the operator gives a directional correction or action command, obey it immediately.\n"
    "Respond as JSON: {\"action\": <action>, \"say\": <text>}.\n"
    f"Allowed actions: {', '.join(ACTION_NAMES)}."
)
PERSISTENT_GOALS_PREFIX = "Persistent goals:"


def build_control_system_message() -> dict[str, Any]:
    return {"role": "system", "content": [{"type": "text", "text": CONTROL_SYSTEM_PROMPT}]}


def build_goal_system_message(*, task_text: str, reward_prompt_id: str = "") -> dict[str, Any]:
    task = str(task_text or "").strip()
    if not task:
        task = f"task linked to reward prompt `{reward_prompt_id}`"
    text = (
        f"{PERSISTENT_GOALS_PREFIX}\n"
        f"1) Primary task: {task}.\n"
        "2) Always follow operator commands faithfully; if a command requests repeated "
        "actions, continue until completion.\n"
        "3) If the operator asks a question, provide a direct spoken answer in `say`."
    )
    return {"role": "system", "content": [{"type": "text", "text": text}]}


def is_goal_system_message(message: dict[str, Any]) -> bool:
    if message.get("role") != "system":
        return False
    for item in _normalize_content(message.get("content")):
        if item.get("type") == "text" and str(item.get("text") or "").startswith(PERSISTENT_GOALS_PREFIX):
            return True
    return False


def build_operator_status_message(
    *,
    user_texts: list[str],
    step_index: int,
    last_reward: float,
    current_reward: float,
    reward_prompt_id: str,
) -> dict[str, Any]:
    speech = "\n".join(text for text in user_texts if str(text).strip()) or "<none>"
    instruction_block = ""
    if speech != "<none>":
        instruction_block = (
            "\n\nOperator command priority:\n"
            "Treat the operator speech above as the highest-priority instruction for this step.\n"
            "`turn left`, `drive left`, or `go left` means choose `drive-left`.\n"
            "`turn right`, `drive right`, or `go right` means choose `drive-right`.\n"
            "`go forward` or `drive forward` means choose `drive-forward`.\n"
            "`go back` or `drive backward` means choose `drive-backward`.\n"
            "`look left`, `look right`, `look up`, and `look forward` map to matching look actions.\n"
            "If the operator asks a question, answer directly in `say`."
        )
    text = (
        f"Operator speech:\n{speech}\n\n"
        "Environment status:\n"
        f"step_index={int(step_index)}\n"
        f"last_reward={float(last_reward):.3f}\n"
        f"current_reward={float(current_reward):.3f}\n"
        f"reward_prompt_id={reward_prompt_id}"
        f"{instruction_block}"
    )
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def build_assistant_action_message(*, action_name: str, generated_text: str = "") -> dict[str, Any]:
    payload = {"action": str(action_name), "say": str(generated_text or "")}
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=True)}],
    }


def ensure_latest_user_image_placeholder(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = [copy_message(message) for message in messages]
    latest_user_index = None
    for idx, message in enumerate(copied):
        if message.get("role") == "user":
            latest_user_index = idx
    if latest_user_index is None:
        copied.append({"role": "user", "content": [{"type": "image"}]})
        return copied
    content = _normalize_content(copied[latest_user_index].get("content"))
    if not any(item.get("type") == "image" for item in content):
        content.insert(0, {"type": "image"})
    copied[latest_user_index]["content"] = content
    return copied


def copy_message(message: dict[str, Any]) -> dict[str, Any]:
    return {"role": message.get("role"), "content": [dict(item) for item in _normalize_content(message.get("content"))]}


def message_with_single_text(message: dict[str, Any], text: str) -> dict[str, Any]:
    return {"role": message.get("role"), "content": [{"type": "text", "text": str(text)}]}


def text_from_message(message: dict[str, Any]) -> str:
    return "\n".join(
        str(item.get("text") or "")
        for item in _normalize_content(message.get("content"))
        if item.get("type") == "text"
    ).strip()


def _normalize_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return [dict(item) for item in content if isinstance(item, dict)]
    return []
