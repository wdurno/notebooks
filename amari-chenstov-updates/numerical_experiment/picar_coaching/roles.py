"""Short frozen-Qwen action requests with explicit operator precedence."""

import hashlib
import time

import torch
from torch.nn import functional as F

from numerical_experiment.picar.dense_roles import prepare_conversation, parse_decision
from numerical_experiment.picar.role_repair import native_text
from numerical_experiment.picar.teacher import prepare_teacher_distribution


TEACHER_PROMPT = """Control the wheeled car. Return ONLY JSON with one key: {"action":"NAME"}.
Do not explain or include say. Choose one of: drive-left, drive-right,
drive-forward, drive-backward, look-left, look-right, look-up, look-forward.
The current operator correction has priority over your visual navigation guess.
Drive-left drives forward while steering the CAR left. Drive-right drives
forward while steering the CAR right. Drive-forward is straight forward.
Drive-backward reverses straight. Look actions STOP driving and move only
the CAMERA; look-forward stops and centers the camera. Looking is not turning.
Instruction "steer left while driving forward" -> {"action":"drive-left"}.
Instruction "begin steering right around the barrier" -> {"action":"drive-right"}.
Instruction "Pass the barrier on the right; begin steering right while driving forward."
-> {"action":"drive-right"}. Do not replace this right turn with drive-forward.
Instruction "Pass the barrier on the left; begin steering left while driving forward."
-> {"action":"drive-left"}.
Instruction "stop driving now" -> {"action":"look-forward"}.
Instruction "reverse straight backward briefly" -> {"action":"drive-backward"}.
Follow the latest active correction even if the ball is already visible.
When no correction is active, approach the red ball until contact. Avoid walls
and blue barriers; turn around an obstruction. Keep the ball roughly centered
when the route is clear. Completed, expired and superseded corrections are history.
"""


class CoachingTeacher:
    def __init__(self, encoder):
        self.encoder = encoder
        self.provenance = {"kind": "fix6-action-only-teacher-v3", "role": "teacher",
            "prompt": TEACHER_PROMPT, "prompt_sha256": hashlib.sha256(TEACHER_PROMPT.encode()).hexdigest(),
            "max_new_tokens": 24, "frames": 1, "max_pixels": 19200,
            "target_kind": "hard_engineered", "control_mapping": "unchanged_probability_weighted"}
        self.last_diagnostic = {}

    @torch.no_grad()
    def distribution(self, history, *, probability_floor):
        tick = time.monotonic()
        with self.encoder._lock:
            inputs = prepare_conversation(self.encoder, history, TEACHER_PROMPT, frames=1, max_pixels=19200)
            texts = native_text(self.encoder, inputs, role="teacher", max_new_tokens=24)
        choices = [parse_decision(text)[0] for text in texts]
        self.last_diagnostic = {"raw_text": texts, "actions": choices,
            "input_tokens": inputs["attention_mask"].sum(-1).tolist(), "seconds": time.monotonic() - tick}
        return prepare_teacher_distribution(F.one_hot(torch.tensor(choices), 8).float(),
            probability_floor=probability_floor, target_kind="hard_engineered")
