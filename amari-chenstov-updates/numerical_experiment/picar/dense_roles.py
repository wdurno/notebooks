"""Fix 2 frozen Qwen roles: generated scores and conversational actions.

Prompt provenance: kl-projection src/picar_kl/reward/prompts.py and
context/messages.py at b0bdccae09ebb60ddabbf30fa63c87da20007150. No external
checkout is imported. Scores are visual rewards, not completion probabilities.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time

import torch
from torch.nn import functional as F

from .models import QwenContextEncoder, render_context, validate_batch
from .role_repair import native_text
from .teacher import prepare_teacher_distribution
from src.robotics.actions import ACTION_NAMES


LEGACY_REWARD_PROMPT = """You are a reinforcement learning assistant in charge of deciding rewards.
You are receiving images from a mobile, robotic camera.
Primary visual task: finding a red ball.
If you see a red ball in the image, return a reward in [1, 10], otherwise zero.
If the ball is far away, return a 1.
If the ball is close enough to fill the screen while still being fully visible, return a 10.
If the ball is too close, entirely filling the screen, return a 1.
For intermediary distances, return interpolated values in (1, 10).
So, the robot only gets the highest score when the red ball is the right distance from the camera.
Return JSON with exactly one key: `reward`.
Example output when no red ball is visible:
{"reward": 0}
Example output when the red ball is centered and at a good distance:
{"reward": 9}"""

CONTACT_REWARD_PROMPT = """You score visual progress for a small mobile robot approaching a red ball.
Use only the image. Blue blocks and walls are not the target.
Return 0 if no red ball is visible. Return 1 for a small far-away ball.
As the visible ball gets larger and closer, increase the reward smoothly toward 10.
A large nearby red ball merits 10, including when partly cropped by the image edge.
Do not require stopping, avoiding an obstacle, or proving physical contact to score visual progress.
Physical success and unsafe collisions are scored separately by the environment.
Return only JSON with exactly one numeric key: {"reward": 0}. Values must be in [0,10]."""

CONTROL_PROMPT = """Control the PiCar robot.
Choose one action and speak one short sentence in `say`.
Authority order: current operator corrections, current images, persistent task, then prior robot messages/actions.
Your previous `say` messages are tentative reports, not facts. Follow new instructions after past actions.
Completed or superseded instructions are history; do not repeat them as fresh commands.
Primary task: drive toward the red ball until contact, while avoiding blue obstacles and walls.
Respond only as JSON: {"action": "NAME", "say": "short status"}.
Allowed actions: drive-left, drive-right, drive-forward, drive-backward, look-left, look-right, look-up, look-forward.
For an explicit stop request, use look-forward (zero drive and steering).
"""

GROUNDED_CONTROL_PROMPT = CONTROL_PROMPT + """
Images come from the car's forward camera. Drive-left moves forward steering left;
drive-right moves forward steering right. Drive-forward goes straight; drive-backward reverses.
Look actions stop driving and rotate the camera; they do not turn the car.
Look-forward stops driving and centres the camera. Commands last one control interval.
Prefer a clear route around an obstacle; do not drive into blue blocks or walls.
When a ball is visibly to the left/right, steer the car toward it, then straighten as it centres.
Executed controls are [camera pan radians, tilt radians, steering radians, drive metres/second].
Pan left and tilt up are positive; steering right is positive. Camera angles are absolute.
"""


def _strict_json(text):
    if not isinstance(text, str):
        raise ValueError("role output must be text")
    text = text.strip()
    if text.startswith("```"):
        wrapped = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
        if wrapped is None:
            raise ValueError("expected one complete JSON code fence and no outside content")
        text = wrapped.group(1).strip()
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("role did not produce JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("role JSON must be an object")
    return value


def parse_reward(text):
    value = _strict_json(text)
    score = value.get("reward")
    if (set(value) != {"reward"} or isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(score) or not 0 <= score <= 10):
        raise ValueError("reward must be a finite numeric value in [0,10]")
    return float(score)


def parse_decision(text):
    value = _strict_json(text)
    if set(value) not in ({"action"}, {"action", "say"}):
        raise ValueError("decision must contain action and optional say only")
    action = value["action"]
    action = "look-forward" if action == "stop" else action
    if action not in ACTION_NAMES or not isinstance(value.get("say", ""), str):
        raise ValueError("unknown action or nontext status")
    return ACTION_NAMES.index(action), value.get("say", "")


def prepare_conversation(encoder, batch, system, *, frames=1, max_pixels=76800, visual_only=False):
    b, t = validate_batch(batch, encoder.context_length)
    if type(frames) is not int or not 1 <= frames <= encoder.context_length:
        raise ValueError("invalid frame count")
    texts, images = [], []
    for i in range(b):
        content = []
        for j in range(max(0, t - frames), t):
            content.append({"type": "image"})
            images.append(batch["images"][i, j].detach().cpu())
        context = render_context(batch, i, t - 1, max_chars=encoder.max_context_chars)
        if visual_only:
            request = "Score the image and return JSON."
        else:
            parsed = json.loads(context)
            active = [e["text"] for e in parsed["instructions"] if e["status"] == "active"]
            request = context + "\nCURRENT OPERATOR INSTRUCTIONS:\n" + "\n".join(active)
            request += "\nFollow the most recent instruction when requests conflict. Return action JSON now."
        content.append({"type": "text", "text": request})
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": content}]
        texts.append(encoder.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    inputs = dict(encoder.processor(text=texts, images=images, padding=True, return_tensors="pt",
                                   do_rescale=False, max_pixels=max_pixels))
    if inputs["input_ids"].shape[1] > encoder.max_tokens:
        raise ValueError("role context exceeds max_tokens; no silent truncation")
    return inputs


class GeneratedVisualReward:
    """Raw 0--10 scores from the explicitly frozen evaluator bank."""

    def __init__(self, encoder, *, rubric="legacy", max_pixels=76800, retries=1):
        if not isinstance(encoder, QwenContextEncoder) or rubric not in {"legacy", "contact"}:
            raise ValueError("actual Qwen and a declared rubric required")
        if type(retries) is not int or not 0 <= retries <= 2:
            raise ValueError("retries must be bounded")
        self.encoder, self.rubric, self.max_pixels, self.retries = encoder, rubric, max_pixels, retries
        self.prompt = LEGACY_REWARD_PROMPT if rubric == "legacy" else CONTACT_REWARD_PROMPT
        self.provenance = {"kind": "generated-visual-score-v1", "role": "evaluator", "rubric": rubric,
            "prompt": self.prompt, "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "max_pixels": max_pixels, "range": [0, 10], "retries": retries, "frames": 1,
            "syntax": "bare-or-single-json-fence-v1"}
        self.last_diagnostic = {}

    @torch.no_grad()
    def scores(self, batch):
        started, attempts = time.monotonic(), []
        with self.encoder._lock:
            inputs = prepare_conversation(self.encoder, batch, self.prompt,
                max_pixels=self.max_pixels, visual_only=True)
            for attempt in range(self.retries + 1):
                texts = native_text(self.encoder, inputs, role="evaluator", max_new_tokens=32)
                record = {"attempt": attempt, "raw_text": texts}
                attempts.append(record)
                try:
                    values = [parse_reward(text) for text in texts]
                except ValueError as exc:
                    record["error"] = str(exc)
                    if attempt == self.retries:
                        self.last_diagnostic = {"attempts": attempts, "seconds": time.monotonic() - started}
                        raise
                else:
                    self.last_diagnostic = {"attempts": attempts, "seconds": time.monotonic() - started,
                        "input_tokens": inputs["attention_mask"].sum(-1).tolist(),
                        "scores": values, "valid": True}
                    return torch.tensor(values, dtype=torch.float32)


class ConversationalTeacher:
    def __init__(self, encoder, *, variant="ported", max_pixels=76800, frames=1):
        if not isinstance(encoder, QwenContextEncoder) or variant not in {"ported", "grounded"}:
            raise ValueError("actual Qwen and versioned teacher required")
        self.encoder, self.variant, self.max_pixels, self.frames = encoder, variant, max_pixels, frames
        self.prompt = CONTROL_PROMPT if variant == "ported" else GROUNDED_CONTROL_PROMPT
        self.provenance = {"kind": "conversational-teacher-v1", "variant": variant, "role": "teacher",
            "prompt": self.prompt, "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "max_pixels": max_pixels, "frames": frames, "target_kind": "hard_engineered"}
        self.last_diagnostic = {}

    @torch.no_grad()
    def distribution(self, batch, *, probability_floor):
        started = time.monotonic()
        with self.encoder._lock:
            inputs = prepare_conversation(self.encoder, batch, self.prompt,
                                          frames=self.frames, max_pixels=self.max_pixels)
            texts = native_text(self.encoder, inputs, role="teacher", max_new_tokens=64)
            self.last_diagnostic = {"raw_text": texts, "seconds": time.monotonic() - started}
            parsed = [parse_decision(text) for text in texts]
            self.last_diagnostic.update(actions=[x[0] for x in parsed], statuses=[x[1] for x in parsed],
                                        input_tokens=inputs["attention_mask"].sum(-1).tolist())
            probabilities = F.one_hot(torch.tensor([x[0] for x in parsed]), 8).float()
            return prepare_teacher_distribution(probabilities, probability_floor=probability_floor,
                                                target_kind="hard_engineered")
